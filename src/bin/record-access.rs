//! Research-only native small-record benchmark. Not a production archive format.
//! Both codecs reuse one prepared model/context; no Python calls per record.
#[cfg(windows)]
mod windows {
    use bre_pair::{
        codec::{Frozen, MatchScratch, Matcher},
        engine::Config,
        huffman::Encoding,
        huffman_archive::{Block, PreparedDecoder, Totals},
        huffman_dictionary::byte_model,
        record_frame::Format,
        scoring,
        shared_huffman::{RuleLengths, StaticModel},
    };
    use std::{
        ffi::{c_char, c_void, CStr},
        fs::{self, File},
        io::{BufReader, BufWriter, Read, Write},
        path::Path,
        sync::Arc,
        time::Instant,
    };

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(path: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }
    type Context = *mut c_void;
    struct Api {
        module: Context,
        version: unsafe extern "C" fn() -> u32,
        error: unsafe extern "C" fn(usize) -> u32,
        error_name: unsafe extern "C" fn(usize) -> *const c_char,
        train: unsafe extern "C" fn(Context, usize, *const c_void, *const usize, u32) -> usize,
        create_cctx: unsafe extern "C" fn() -> Context,
        create_dctx: unsafe extern "C" fn() -> Context,
        free_cctx: unsafe extern "C" fn(Context) -> usize,
        free_dctx: unsafe extern "C" fn(Context) -> usize,
        create_cdict: unsafe extern "C" fn(*const c_void, usize, i32) -> Context,
        create_ddict: unsafe extern "C" fn(*const c_void, usize) -> Context,
        free_cdict: unsafe extern "C" fn(Context) -> usize,
        free_ddict: unsafe extern "C" fn(Context) -> usize,
        compress: unsafe extern "C" fn(
            Context,
            Context,
            usize,
            *const c_void,
            usize,
            *const c_void,
        ) -> usize,
        decompress: unsafe extern "C" fn(
            Context,
            Context,
            usize,
            *const c_void,
            usize,
            *const c_void,
        ) -> usize,
        bound: unsafe extern "C" fn(usize) -> usize,
        content_size: unsafe extern "C" fn(*const c_void, usize) -> u64,
    }
    impl Api {
        fn load(path: &Path) -> Result<Self, String> {
            use std::os::windows::ffi::OsStrExt;
            let path: Vec<_> = path.as_os_str().encode_wide().chain([0]).collect();
            let module = unsafe { LoadLibraryW(path.as_ptr()) };
            if module.is_null() {
                return Err(std::io::Error::last_os_error().to_string());
            }
            macro_rules! symbol {
                ($name:literal, $signature:ty) => {{
                    let address =
                        unsafe { GetProcAddress(module, concat!($name, "\0").as_ptr().cast()) };
                    if address.is_null() {
                        unsafe {
                            FreeLibrary(module);
                        }
                        return Err(concat!("missing libzstd export: ", $name).into());
                    }
                    // The destination is the exact C signature of the Api field.
                    unsafe { std::mem::transmute::<*mut c_void, $signature>(address) }
                }};
            }
            Ok(Self {
                module,
                version: symbol!("ZSTD_versionNumber", unsafe extern "C" fn() -> u32),
                error: symbol!("ZSTD_isError", unsafe extern "C" fn(usize) -> u32),
                error_name: symbol!(
                    "ZSTD_getErrorName",
                    unsafe extern "C" fn(usize) -> *const c_char
                ),
                train: symbol!(
                    "ZDICT_trainFromBuffer",
                    unsafe extern "C" fn(Context, usize, *const c_void, *const usize, u32) -> usize
                ),
                create_cctx: symbol!("ZSTD_createCCtx", unsafe extern "C" fn() -> Context),
                create_dctx: symbol!("ZSTD_createDCtx", unsafe extern "C" fn() -> Context),
                free_cctx: symbol!("ZSTD_freeCCtx", unsafe extern "C" fn(Context) -> usize),
                free_dctx: symbol!("ZSTD_freeDCtx", unsafe extern "C" fn(Context) -> usize),
                create_cdict: symbol!(
                    "ZSTD_createCDict",
                    unsafe extern "C" fn(*const c_void, usize, i32) -> Context
                ),
                create_ddict: symbol!(
                    "ZSTD_createDDict",
                    unsafe extern "C" fn(*const c_void, usize) -> Context
                ),
                free_cdict: symbol!("ZSTD_freeCDict", unsafe extern "C" fn(Context) -> usize),
                free_ddict: symbol!("ZSTD_freeDDict", unsafe extern "C" fn(Context) -> usize),
                compress: symbol!(
                    "ZSTD_compress_usingCDict",
                    unsafe extern "C" fn(
                        Context,
                        Context,
                        usize,
                        *const c_void,
                        usize,
                        *const c_void,
                    ) -> usize
                ),
                decompress: symbol!(
                    "ZSTD_decompress_usingDDict",
                    unsafe extern "C" fn(
                        Context,
                        Context,
                        usize,
                        *const c_void,
                        usize,
                        *const c_void,
                    ) -> usize
                ),
                bound: symbol!("ZSTD_compressBound", unsafe extern "C" fn(usize) -> usize),
                content_size: symbol!(
                    "ZSTD_getFrameContentSize",
                    unsafe extern "C" fn(*const c_void, usize) -> u64
                ),
            })
        }
        fn check(&self, code: usize) -> Result<usize, String> {
            if unsafe { (self.error)(code) } == 0 {
                Ok(code)
            } else {
                Err(unsafe { CStr::from_ptr((self.error_name)(code)) }
                    .to_string_lossy()
                    .into_owned())
            }
        }
    }
    impl Drop for Api {
        fn drop(&mut self) {
            unsafe {
                FreeLibrary(self.module);
            }
        }
    }
    trait Codec {
        fn encode(&mut self, row: &[u8]) -> Result<Vec<u8>, String>;
        fn decode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String>;
        fn last_header_bytes(&self) -> Option<usize> {
            None
        }
        fn framing(&self) -> Option<Format> {
            None
        }
    }
    struct Zstd<'a> {
        api: &'a Api,
        cctx: Context,
        dctx: Context,
        cdict: Context,
        ddict: Context,
    }
    impl<'a> Zstd<'a> {
        fn new(api: &'a Api, dictionary: &[u8], level: i32) -> Result<Self, String> {
            let model = Self {
                api,
                cctx: unsafe { (api.create_cctx)() },
                dctx: unsafe { (api.create_dctx)() },
                cdict: unsafe {
                    (api.create_cdict)(dictionary.as_ptr().cast(), dictionary.len(), level)
                },
                ddict: unsafe { (api.create_ddict)(dictionary.as_ptr().cast(), dictionary.len()) },
            };
            if [model.cctx, model.dctx, model.cdict, model.ddict]
                .iter()
                .any(|p| p.is_null())
            {
                return Err("libzstd allocation failed".into());
            }
            Ok(model)
        }
    }
    impl Drop for Zstd<'_> {
        fn drop(&mut self) {
            unsafe {
                (self.api.free_cctx)(self.cctx);
                (self.api.free_dctx)(self.dctx);
                (self.api.free_cdict)(self.cdict);
                (self.api.free_ddict)(self.ddict);
            }
        }
    }
    impl Codec for Zstd<'_> {
        fn encode(&mut self, row: &[u8]) -> Result<Vec<u8>, String> {
            let mut out = vec![0; unsafe { (self.api.bound)(row.len()) }];
            let n = self.api.check(unsafe {
                (self.api.compress)(
                    self.cctx,
                    out.as_mut_ptr().cast(),
                    out.len(),
                    row.as_ptr().cast(),
                    row.len(),
                    self.cdict,
                )
            })?;
            out.truncate(n);
            Ok(out)
        }
        fn decode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
            let n = unsafe { (self.api.content_size)(frame.as_ptr().cast(), frame.len()) };
            if n > 1 << 20 {
                return Err("invalid or oversized benchmark record".into());
            }
            out.resize(n as usize, 0);
            let got = self.api.check(unsafe {
                (self.api.decompress)(
                    self.dctx,
                    out.as_mut_ptr().cast(),
                    out.len(),
                    frame.as_ptr().cast(),
                    frame.len(),
                    self.ddict,
                )
            })?;
            if got != out.len() {
                return Err("zstd record length differs".into());
            }
            Ok(())
        }
    }
    trait RecordModel {
        fn prepare(
            &self,
        ) -> Result<(Matcher, Arc<bre_pair::huffman::Codebook>, PreparedDecoder), String>;
        fn decode(
            &self,
            decoder: &PreparedDecoder,
            block: &Block,
            out: &mut Vec<u8>,
        ) -> Result<bool, String>;
        fn decode_field(
            &self,
            decoder: &PreparedDecoder,
            payload: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(u64, u64), String>;
    }
    impl RecordModel for Frozen {
        fn prepare(
            &self,
        ) -> Result<(Matcher, Arc<bre_pair::huffman::Codebook>, PreparedDecoder), String> {
            let matcher = Frozen::prepare(self)?;
            let book = Arc::new(bre_pair::huffman::Codebook::from_lengths(
                &self.book.lengths().collect::<Vec<_>>(),
            )?);
            let decoder = PreparedDecoder::with_codebook(
                &self.dictionary,
                Arc::clone(&book),
                false,
                64 << 20,
            )?;
            Ok((matcher, book, decoder))
        }
        fn decode(
            &self,
            decoder: &PreparedDecoder,
            block: &Block,
            out: &mut Vec<u8>,
        ) -> Result<bool, String> {
            decoder.decode_block(&self.dictionary, block, out)
        }
        fn decode_field(
            &self,
            decoder: &PreparedDecoder,
            payload: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(u64, u64), String> {
            decoder.decode_field(&self.dictionary, payload, 1 << 20, out)
        }
    }
    impl RecordModel for StaticModel {
        fn prepare(
            &self,
        ) -> Result<(Matcher, Arc<bre_pair::huffman::Codebook>, PreparedDecoder), String> {
            Ok((
                StaticModel::prepare(self)?,
                self.codebook(),
                self.prepare_decoder(false, 64 << 20)?,
            ))
        }
        fn decode(
            &self,
            decoder: &PreparedDecoder,
            block: &Block,
            out: &mut Vec<u8>,
        ) -> Result<bool, String> {
            self.decode_block(decoder, block, out)
        }
        fn decode_field(
            &self,
            decoder: &PreparedDecoder,
            payload: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(u64, u64), String> {
            StaticModel::decode_field(self, decoder, payload, 1 << 20, out)
        }
    }
    struct Brepair<M> {
        model: M,
        matcher: Matcher,
        scratch: MatchScratch,
        terminals: Vec<u32>,
        encoder: Encoding,
        decoder: PreparedDecoder,
        block: Block,
        format: Format,
        header_bytes: usize,
        eof_decoder: bool,
    }
    impl<M: RecordModel> Brepair<M> {
        fn new(model: M) -> Result<Self, String> {
            let (matcher, book, decoder) = model.prepare()?;
            let encoder = book.encoding();
            Ok(Self {
                model,
                matcher,
                scratch: Default::default(),
                terminals: Vec::new(),
                encoder,
                decoder,
                format: Format::Fixed20,
                header_bytes: 20,
                eof_decoder: false,
                block: Block {
                    book,
                    payload: Vec::new(),
                    bits: 0,
                    totals: Totals::default(),
                },
            })
        }
    }
    impl<M: RecordModel> Codec for Brepair<M> {
        fn encode(&mut self, row: &[u8]) -> Result<Vec<u8>, String> {
            self.terminals.clear();
            self.terminals.extend(row.iter().map(|&b| u32::from(b)));
            self.terminals.push(256);
            let symbols = self
                .matcher
                .segment_with(&self.terminals, &mut self.scratch)?;
            let (payload, bits) = self.encoder.encode(symbols)?;
            let frame =
                self.format
                    .encode(row.len() as u64, bits, symbols.len() as u64, &payload)?;
            self.header_bytes = frame.len() - payload.len();
            Ok(frame)
        }
        fn decode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
            let record = self.format.parse(frame)?;
            if self.eof_decoder || !record.has_counters {
                let (bits, symbols) =
                    self.model
                        .decode_field(&self.decoder, record.payload, out)?;
                if record.has_counters
                    && (record.bytes != out.len() as u64
                        || record.bits != bits
                        || record.symbols != symbols)
                {
                    return Err("EOF-decoded totals differ from fixed counter control".into());
                }
                return Ok(());
            }
            let (bytes, bits, symbols) = (record.bytes, record.bits, record.symbols);
            self.block.payload.clear();
            self.block.payload.extend_from_slice(record.payload);
            self.block.bits = bits;
            self.block.totals = Totals {
                bytes,
                fields: 1,
                tokens: bytes + 1,
                symbols,
            };
            out.clear();
            if !self.model.decode(&self.decoder, &self.block, out)? {
                return Err("benchmark row lacks EOF".into());
            }
            Ok(())
        }
        fn last_header_bytes(&self) -> Option<usize> {
            Some(self.header_bytes)
        }
        fn framing(&self) -> Option<Format> {
            Some(self.format)
        }
    }
    fn mix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    fn offsets(bytes: &[u8]) -> Vec<usize> {
        let mut out = vec![0];
        out.extend(
            bytes
                .iter()
                .enumerate()
                .filter_map(|(i, &b)| (b == b'\n').then_some(i + 1)),
        );
        if out.last() != Some(&bytes.len()) {
            out.push(bytes.len());
        }
        out
    }
    fn benchmark(
        name: &str,
        codec: &mut impl Codec,
        model_bytes: u64,
        raw: &[u8],
        rows: &[usize],
        folder: &Path,
    ) -> Result<String, String> {
        benchmark_records(name, codec, model_bytes, raw, rows, folder, None)
    }
    fn benchmark_records(
        name: &str,
        codec: &mut impl Codec,
        model_bytes: u64,
        raw: &[u8],
        rows: &[usize],
        folder: &Path,
        reframe: Option<(&Path, &Path, Format)>,
    ) -> Result<String, String> {
        let ioerr = |e: std::io::Error| e.to_string();
        if rows.len() < 2 {
            return Err("record benchmark requires nonempty test data".into());
        }
        let cached = if let Some((frames, index, _)) = reframe {
            let frames = fs::read(frames).map_err(ioerr)?;
            let bytes = fs::read(index).map_err(ioerr)?;
            if bytes.len() != rows.len() * 8 {
                return Err("cached record index length differs".into());
            }
            let index: Vec<_> = bytes
                .chunks_exact(8)
                .map(|b| {
                    usize::try_from(u64::from_le_bytes(b.try_into().unwrap()))
                        .map_err(|_| "record index overflow")
                })
                .collect::<Result<_, _>>()?;
            if index[0] != 0
                || index.last() != Some(&frames.len())
                || index.windows(2).any(|w| w[0] >= w[1])
            {
                return Err("invalid cached record index".into());
            }
            Some((frames, index))
        } else {
            None
        };
        let mut data =
            BufWriter::new(File::create_new(folder.join(format!("{name}.frames"))).map_err(ioerr)?);
        let global_header = codec.framing().map_or_else(Vec::new, Format::file_header);
        data.write_all(&global_header).map_err(ioerr)?;
        let mut index = Vec::with_capacity(rows.len());
        index.push(global_header.len());
        let (mut header_bytes, mut has_header) = (0u64, false);
        let start = Instant::now();
        for (i, w) in rows.windows(2).enumerate() {
            let (frame, header) = if let Some((data, positions)) = &cached {
                let old = Format::Fixed20.parse(&data[positions[i]..positions[i + 1]])?;
                if old.bytes as usize != w[1] - w[0] {
                    return Err("cached record length differs".into());
                }
                let frame =
                    reframe
                        .unwrap()
                        .2
                        .encode(old.bytes, old.bits, old.symbols, old.payload)?;
                let header = frame.len() - old.payload.len();
                (frame, Some(header))
            } else {
                let frame = codec.encode(&raw[w[0]..w[1]])?;
                let header = codec.last_header_bytes();
                (frame, header)
            };
            if let Some(header) = header {
                header_bytes += header as u64;
                has_header = true;
            }
            data.write_all(&frame).map_err(ioerr)?;
            index.push(index.last().unwrap() + frame.len());
            if i % 1_000_000 == 999_999 {
                eprintln!("{name}: encoded {} rows", i + 1);
            }
        }
        data.flush().map_err(ioerr)?;
        drop(data);
        let encode_seconds = start.elapsed().as_secs_f64();
        drop(cached);
        let mut idx = BufWriter::new(
            File::create_new(folder.join(format!("{name}.offsets.u64"))).map_err(ioerr)?,
        );
        for &at in &index {
            idx.write_all(&(at as u64).to_le_bytes()).map_err(ioerr)?;
        }
        idx.flush().map_err(ioerr)?;
        drop(idx);
        let encoded = fs::read(folder.join(format!("{name}.frames"))).map_err(ioerr)?;
        if encoded.get(..global_header.len()) != Some(global_header.as_slice()) {
            return Err("global frame header differs".into());
        }
        let mut out = Vec::new();
        let start = Instant::now();
        for (i, w) in index.windows(2).enumerate() {
            codec.decode(&encoded[w[0]..w[1]], &mut out)?;
            if out != raw[rows[i]..rows[i + 1]] {
                return Err(format!("{name}: round trip differs at row {i}"));
            }
        }
        let verified_decode_seconds = start.elapsed().as_secs_f64();
        // Identical deterministic queries; all frames/model resident in RAM.
        let queries: Vec<_> = (0..20000u64)
            .map(|i| mix(i ^ 20261106) as usize % (rows.len() - 1))
            .collect();
        for &i in queries.iter().take(1000) {
            codec.decode(&encoded[index[i]..index[i + 1]], &mut out)?;
        }
        let mut latency = Vec::with_capacity(queries.len());
        for &i in &queries {
            let start = Instant::now();
            codec.decode(&encoded[index[i]..index[i + 1]], &mut out)?;
            latency.push(start.elapsed().as_nanos() as u64);
            if out != raw[rows[i]..rows[i + 1]] {
                return Err("random row mismatch".into());
            }
        }
        latency.sort_unstable();
        let percentile = |p: usize| latency[(latency.len() * p).div_ceil(100) - 1];
        let payload = encoded.len() as u64;
        let index_bytes = index.len() as u64 * 8;
        let (encode_json, reframe_json) = if reframe.is_some() {
            ("null".to_owned(), format!("{encode_seconds:.6}"))
        } else {
            (format!("{encode_seconds:.6}"), "null".to_owned())
        };
        let headers = if has_header {
            header_bytes.to_string()
        } else {
            "null".into()
        };
        Ok(format!(concat!("{{\"codec\":\"{}\",\"rows\":{},\"original_bytes\":{},\"model_bytes\":{},",
            "\"frames_bytes\":{},\"index_bytes\":{},\"total_stored_bytes\":{},\"encode_seconds\":{},\"reframe_seconds\":{},\"record_header_bytes\":{},\"global_frame_header_bytes\":{},",
            "\"verified_sequential_decode_seconds\":{:.6},\"random_queries\":{},\"random_p50_ns\":{},",
            "\"random_p95_ns\":{},\"random_p99_ns\":{},\"all_rows_byte_verified\":true}}"),
            name,rows.len()-1,raw.len(),model_bytes,payload,index_bytes,model_bytes+payload+index_bytes,
            encode_json,reframe_json,headers,global_header.len(),verified_decode_seconds,queries.len(),percentile(50),percentile(95),percentile(99)))
    }
    fn tagged(mut report: String, fields: &str) -> String {
        report.pop();
        report.push(',');
        report.push_str(fields);
        report.push('}');
        report
    }
    fn files_equal(a: &Path, b: &Path) -> Result<bool, String> {
        if fs::metadata(a).map_err(|e| e.to_string())?.len()
            != fs::metadata(b).map_err(|e| e.to_string())?.len()
        {
            return Ok(false);
        }
        let mut a = BufReader::new(File::open(a).map_err(|e| e.to_string())?);
        let mut b = BufReader::new(File::open(b).map_err(|e| e.to_string())?);
        let mut left = [0u8; 65536];
        let mut right = [0u8; 65536];
        loop {
            let n = a.read(&mut left).map_err(|e| e.to_string())?;
            if n == 0 {
                return Ok(true);
            }
            b.read_exact(&mut right[..n]).map_err(|e| e.to_string())?;
            if left[..n] != right[..n] {
                return Ok(false);
            }
        }
    }
    fn experiments(args: &[String]) -> Result<(), String> {
        let usage="record-access TRAIN TEST NEW_DIRECTORY LIBZSTD.dll --baseline-model MODEL.brc [--refits 2 --zstd-controls all|matched --reference-frames FILE]";
        if args.len() < 6 {
            return Err(usage.into());
        }
        let (mut baseline_path, mut reference) = (None, None);
        let mut refits = 2usize;
        let mut all_zstd = true;
        for pair in args[4..].chunks(2) {
            let value = pair.get(1).ok_or("missing record experiment option")?;
            match pair[0].as_str() {
                "--baseline-model" => baseline_path = Some(Path::new(value)),
                "--refits" => refits = value.parse().map_err(|_| "invalid refit count")?,
                "--zstd-controls" => match value.as_str() {
                    "all" => all_zstd = true,
                    "matched" => all_zstd = false,
                    _ => return Err("use all or matched Zstd controls".into()),
                },
                "--reference-frames" => reference = Some(Path::new(value)),
                _ => return Err(usage.into()),
            }
        }
        if refits > 16 {
            return Err("refits must be <=16".into());
        }
        let baseline = fs::read(baseline_path.ok_or("baseline model is required")?)
            .map_err(|e| e.to_string())?;
        let model = Frozen::from_bytes(&baseline)?;
        let folder = Path::new(&args[2]);
        fs::create_dir(folder).map_err(|e| e.to_string())?;
        let api = Api::load(Path::new(&args[3]))?;
        if unsafe { (api.version)() } != 10507 {
            return Err("requires libzstd 1.5.7".into());
        }
        eprintln!("Fitting {refits} exact training-only reparses on the saved grammar; shared data/reference weights");
        let fitted = bre_pair::record_refit::training(&model, Path::new(&args[0]), refits, true)?;
        if model.dictionary.to_bytes() != fitted.model.dictionary.to_bytes() {
            return Err("refit changed the grammar".into());
        }
        let refitted = fitted.model.to_bytes()?;
        let compact0 = StaticModel::encode_frozen(&model, RuleLengths::Auto)?;
        let compact2 = StaticModel::encode_frozen(&fitted.model, RuleLengths::Auto)?;
        for (name, bytes) in [
            ("baseline.brc", &baseline),
            ("baseline.brhm", &compact0),
            ("refitted.brc", &refitted),
            ("refitted.brhm", &compact2),
        ] {
            fs::write(folder.join(name), bytes).map_err(|e| e.to_string())?;
        }
        // Prepare every dictionary on training rows before opening the holdout.
        let training = fs::read(&args[0]).map_err(|e| e.to_string())?;
        let train_rows = offsets(&training);
        let sizes: Vec<_> = train_rows.windows(2).map(|w| w[1] - w[0]).collect();
        let mut budgets = vec![
            baseline.len(),
            compact0.len(),
            refitted.len(),
            compact2.len(),
        ];
        if all_zstd {
            budgets.extend([0, 65536, 262144]);
        }
        budgets.sort_unstable();
        budgets.dedup();
        let mut zstd_models = Vec::new();
        for capacity in budgets {
            eprintln!("Training Zstd dictionary with capacity {capacity}");
            let start = Instant::now();
            let mut dictionary = vec![0u8; capacity];
            if capacity > 0 {
                let n = api.check(unsafe {
                    (api.train)(
                        dictionary.as_mut_ptr().cast(),
                        dictionary.len(),
                        training.as_ptr().cast(),
                        sizes.as_ptr(),
                        u32::try_from(sizes.len()).map_err(|_| "too many training rows")?,
                    )
                })?;
                dictionary.truncate(n);
            }
            let seconds = start.elapsed().as_secs_f64();
            fs::write(folder.join(format!("zstd-{capacity}.dict")), &dictionary)
                .map_err(|e| e.to_string())?;
            zstd_models.push((capacity, dictionary, seconds));
        }
        drop(training);
        drop(train_rows);
        drop(sizes);
        let points=fitted.points.iter().map(|p|format!("{{\"iteration\":{},\"payload_bits\":{},\"padded_payload_bytes\":{},\"fields\":{},\"symbols\":{},\"legacy_model_bytes\":{},\"compact_model_bytes\":{},\"seconds\":{:.6}}}",
            p.iteration,p.payload_bits,p.payload_bytes,p.fields,p.symbols,p.legacy_model_bytes,p.compact_model_bytes,p.seconds)).collect::<Vec<_>>().join(",");
        let protocol=format!(concat!("{{\"workload\":\"TPC-H-derived independent held-out rows; not SQL performance\",",
            "\"resident_random_access\":true,\"workers_during_measurement\":1,\"query_seed\":20261106,",
            "\"training_only_models\":true,\"models_frozen_before_opening_holdout\":true,\"zstd_version\":10507,",
            "\"refits\":{},\"refit_weights\":\"data plus stored dictionary references\",\"training_points\":[{}],",
            "\"framing_reuses_identical_coded_payloads\":true,\"framing_only_encode_seconds_are_null\":true,",
            "\"sequential_timing_includes_byte_comparison\":true,\"results\":["),refits,points);
        fs::write(
            folder.join("model-preparation.json"),
            format!("{protocol}]}}\n"),
        )
        .map_err(|e| e.to_string())?;
        let raw = fs::read(&args[1]).map_err(|e| e.to_string())?;
        let rows = offsets(&raw);
        let mut results = Vec::new();
        let save = |results: &[String]| {
            fs::write(
                folder.join("report.json"),
                format!("{protocol}{}]}}\n", results.join(",")),
            )
            .map_err(|e| e.to_string())
        };
        for (name, legacy, compact) in [
            ("baseline", &baseline, &compact0),
            ("refitted", &refitted, &compact2),
        ] {
            eprintln!("Measuring {name} legacy model with fixed20 framing");
            let mut codec = Brepair::new(Frozen::from_bytes(legacy)?)?;
            let legacy_name = format!("brepair-{name}-legacy-fixed20");
            let measured = benchmark(
                &legacy_name,
                &mut codec,
                legacy.len() as u64,
                &raw,
                &rows,
                folder,
            )?;
            results.push(tagged(measured,&format!("\"model\":\"{name}\",\"representation\":\"legacy\",\"framing\":\"fixed20\",\"per_record_checksum\":true")));
            save(&results)?;
            drop(codec);
            if name == "baseline" {
                if let Some(reference) = reference {
                    if !files_equal(&folder.join(format!("{legacy_name}.frames")), reference)? {
                        return Err("baseline frames differ from saved experiment".into());
                    }
                }
            }
            eprintln!("Measuring {name} compact model with unchanged fixed20 framing");
            let mut codec = Brepair::new(StaticModel::from_bytes(compact)?)?;
            let fixed_name = format!("brepair-{name}-compact-fixed20");
            let measured = benchmark(
                &fixed_name,
                &mut codec,
                compact.len() as u64,
                &raw,
                &rows,
                folder,
            )?;
            for suffix in ["frames", "offsets.u64"] {
                if !files_equal(
                    &folder.join(format!("{legacy_name}.{suffix}")),
                    &folder.join(format!("{fixed_name}.{suffix}")),
                )? {
                    return Err("pure model serialization changed held-out coding".into());
                }
            }
            results.push(tagged(measured,&format!("\"model\":\"{name}\",\"representation\":\"compact\",\"framing\":\"fixed20\",\"per_record_checksum\":true,\"identical_legacy_frames\":true")));
            save(&results)?;
            for format in [Format::VarintChecked, Format::VarintPlain] {
                codec.format = format;
                let framed_name = format!("brepair-{name}-compact-{}", format.name());
                let frames = folder.join(format!("{fixed_name}.frames"));
                let index = folder.join(format!("{fixed_name}.offsets.u64"));
                eprintln!("Reframing identical {name} payloads with {}", format.name());
                let measured = benchmark_records(
                    &framed_name,
                    &mut codec,
                    compact.len() as u64,
                    &raw,
                    &rows,
                    folder,
                    Some((&frames, &index, format)),
                )?;
                results.push(tagged(measured,&format!("\"model\":\"{name}\",\"representation\":\"compact\",\"framing\":\"{}\",\"per_record_checksum\":{}",format.name(),format.checked())));
                save(&results)?;
            }
        }
        for (capacity, dictionary, training_seconds) in zstd_models {
            let start = Instant::now();
            let mut codec = Zstd::new(&api, &dictionary, 3)?;
            let preparation = start.elapsed().as_secs_f64();
            eprintln!("Measuring Zstd-3 with {capacity}-byte dictionary budget");
            let name = format!("zstd3-{capacity}");
            let measured = benchmark(
                &name,
                &mut codec,
                dictionary.len() as u64,
                &raw,
                &rows,
                folder,
            )?;
            results.push(tagged(measured,&format!("\"representation\":\"zstd\",\"framing\":\"standard\",\"per_record_checksum\":false,\"requested_dictionary_bytes\":{capacity},\"dictionary_training_seconds\":{training_seconds:.6},\"preparation_seconds\":{preparation:.6}")));
            save(&results)?;
        }
        println!("{protocol}{}]}}", results.join(","));
        Ok(())
    }
    fn eof_experiments(args: &[String]) -> Result<(), String> {
        if args.len() != 4 {
            return Err(
                "record-access --eof-from PREVIOUS_NATIVE_DIRECTORY TEST.psv NEW_DIRECTORY".into(),
            );
        }
        let ioerr = |e: std::io::Error| e.to_string();
        let source = Path::new(&args[1]);
        let folder = Path::new(&args[3]);
        fs::create_dir(folder).map_err(ioerr)?;
        let raw = fs::read(&args[2]).map_err(ioerr)?;
        let rows = offsets(&raw);
        let protocol = concat!("{\"workload\":\"Same TPC-H held-out rows; EOF framing only, no fitting or segmentation\",",
            "\"models_frozen_before_opening_holdout\":true,\"training_only_models\":true,",
            "\"resident_random_access\":true,\"workers_during_measurement\":1,\"query_seed\":20261106,",
            "\"sequential_timing_includes_byte_comparison\":true,\"results\":[");
        let mut results = Vec::new();
        for name in ["baseline", "refitted"] {
            let model_bytes = fs::read(source.join(format!("{name}.brhm"))).map_err(ioerr)?;
            let model = StaticModel::from_bytes(&model_bytes)?;
            fs::write(folder.join(format!("{name}.brhm")), &model_bytes).map_err(ioerr)?;
            let mut codec = Brepair::new(model)?;
            codec.eof_decoder = true;
            let prefix = format!("brepair-{name}-compact-fixed20");
            let frames = source.join(format!("{prefix}.frames"));
            let index = source.join(format!("{prefix}.offsets.u64"));
            for format in [Format::Fixed20, Format::EofChecked, Format::EofPlain] {
                codec.format = format;
                let case = format!("brepair-{name}-{}-eof-decoder", format.name());
                eprintln!("Measuring {case}: frozen payloads; decode until EOF");
                let measured = benchmark_records(
                    &case,
                    &mut codec,
                    model_bytes.len() as u64,
                    &raw,
                    &rows,
                    folder,
                    Some((&frames, &index, format)),
                )?;
                if format == Format::Fixed20
                    && (!files_equal(&frames, &folder.join(format!("{case}.frames")))?
                        || !files_equal(&index, &folder.join(format!("{case}.offsets.u64")))?)
                {
                    return Err("EOF decoder control changed fixed20 frames/index".into());
                }
                results.push(tagged(measured, &format!("\"model\":\"{name}\",\"representation\":\"compact\",\"framing\":\"{}\",\"per_record_checksum\":{},\"eof_decoder\":true", format.name(), format.checked())));
                fs::write(
                    folder.join("report.json"),
                    format!("{protocol}{}]}}\n", results.join(",")),
                )
                .map_err(ioerr)?;
            }
        }
        println!("{protocol}{}]}}", results.join(","));
        Ok(())
    }
    fn train_zstd_dictionaries(args: &[String]) -> Result<(), String> {
        let ioerr = |e: std::io::Error| e.to_string();
        if args.len() != 6 || args[4] != "--budgets" {
            return Err(
                "record-access --zstd-train TRAIN NEW_DIRECTORY LIBZSTD.dll --budgets N,N,..."
                    .into(),
            );
        }
        let budgets = args[5]
            .split(',')
            .map(|v| v.parse::<usize>().map_err(|_| "invalid dictionary budget"))
            .collect::<Result<Vec<_>, _>>()?;
        if budgets.is_empty()
            || budgets.contains(&0)
            || budgets
                .iter()
                .enumerate()
                .any(|(i, v)| budgets[..i].contains(v))
        {
            return Err("dictionary budgets must be positive and distinct".into());
        }
        let api = Api::load(Path::new(&args[3]))?;
        if unsafe { (api.version)() } != 10507 {
            return Err("requires libzstd 1.5.7".into());
        }
        let training = fs::read(&args[1]).map_err(ioerr)?;
        let rows = offsets(&training);
        let sizes: Vec<_> = rows.windows(2).map(|w| w[1] - w[0]).collect();
        let folder = Path::new(&args[2]);
        fs::create_dir(folder).map_err(ioerr)?;
        let mut results = Vec::new();
        for capacity in budgets {
            eprintln!(
                "Training Zstd dictionary: capacity {capacity}, {} training bytes",
                training.len()
            );
            let mut dictionary = vec![0; capacity];
            let start = Instant::now();
            let size = unsafe {
                (api.train)(
                    dictionary.as_mut_ptr().cast(),
                    dictionary.len(),
                    training.as_ptr().cast(),
                    sizes.as_ptr(),
                    u32::try_from(sizes.len()).map_err(|_| "too many training rows")?,
                )
            };
            match api.check(size) {
                Ok(size) => {
                    dictionary.truncate(size);
                    fs::write(folder.join(format!("zstd-{capacity}.dict")), &dictionary)
                        .map_err(ioerr)?;
                    results.push(format!("{{\"capacity\":{capacity},\"actual_bytes\":{size},\"trained\":true,\"seconds\":{:.6}}}", start.elapsed().as_secs_f64()));
                }
                Err(error) => {
                    eprintln!(
                        "Dictionary capacity {capacity} unsupported by this trainer/data: {error}"
                    );
                    results.push(format!(
                        "{{\"capacity\":{capacity},\"trained\":false,\"seconds\":{:.6}}}",
                        start.elapsed().as_secs_f64()
                    ));
                }
            }
            fs::write(
                folder.join("training.json"),
                format!(
                    "{{\"training_bytes\":{},\"training_rows\":{},\"results\":[{}]}}\n",
                    training.len(),
                    sizes.len(),
                    results.join(",")
                ),
            )
            .map_err(ioerr)?;
        }
        Ok(())
    }
    fn zstd_frozen_experiments(args: &[String]) -> Result<(), String> {
        let ioerr = |e: std::io::Error| e.to_string();
        let usage = "record-access --zstd-from SAVED_DIRECTORY TEST NEW_DIRECTORY LIBZSTD.dll [--levels 3,19,22 --budgets 65536,163791,262144 --max-rows N]";
        if args.len() < 5 {
            return Err(usage.into());
        }
        let (mut levels, mut budgets, mut maximum_rows) = (
            vec![3i32, 19, 22],
            vec![65536usize, 163791, 262144],
            usize::MAX,
        );
        for pair in args[5..].chunks(2) {
            let value = pair.get(1).ok_or("missing Zstd experiment option")?;
            match pair[0].as_str() {
                "--levels" => {
                    levels = value
                        .split(',')
                        .map(|v| v.parse().map_err(|_| "invalid Zstd level"))
                        .collect::<Result<_, _>>()?
                }
                "--budgets" => {
                    budgets = value
                        .split(',')
                        .map(|v| v.parse().map_err(|_| "invalid dictionary budget"))
                        .collect::<Result<_, _>>()?
                }
                "--max-rows" => maximum_rows = value.parse().map_err(|_| "invalid row limit")?,
                _ => return Err(usage.into()),
            }
        }
        if levels.is_empty()
            || levels.iter().any(|&v| !(1..=22).contains(&v))
            || budgets.is_empty()
            || budgets.contains(&0)
            || maximum_rows == 0
        {
            return Err(
                "use levels 1..=22, positive dictionary budgets and a positive row limit".into(),
            );
        }
        let unique = |v: &[usize]| v.iter().enumerate().all(|(i, x)| !v[..i].contains(x));
        if !unique(&budgets)
            || levels
                .iter()
                .enumerate()
                .any(|(i, x)| levels[..i].contains(x))
        {
            return Err("duplicate Zstd level or dictionary budget".into());
        }
        let saved = Path::new(&args[1]);
        // Reuse the exact training-only dictionaries. Loading all dictionaries
        // before the test data also fixes the model selection boundary.
        let dictionaries = budgets
            .iter()
            .map(|&capacity| {
                let bytes = fs::read(saved.join(format!("zstd-{capacity}.dict"))).map_err(ioerr)?;
                if bytes.is_empty() || bytes.len() > capacity {
                    return Err("saved dictionary length differs from its budget".to_string());
                }
                Ok((capacity, bytes))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let folder = Path::new(&args[3]);
        fs::create_dir(folder).map_err(ioerr)?;
        let api = Api::load(Path::new(&args[4]))?;
        if unsafe { (api.version)() } != 10507 {
            return Err("requires libzstd 1.5.7".into());
        }
        let mut raw = fs::read(&args[2]).map_err(ioerr)?;
        let mut rows = offsets(&raw);
        if maximum_rows < rows.len().saturating_sub(1) {
            rows.truncate(maximum_rows + 1);
            raw.truncate(*rows.last().ok_or("no records")?);
        }
        let prefix = concat!("{\"workload\":\"frozen-dictionary independent byte strings; not database table storage\",",
            "\"zstd_version\":10507,\"frozen_dictionary_reuse\":true,\"models_frozen_before_opening_holdout\":true,",
            "\"resident_random_access\":true,\"workers_during_measurement\":1,",
            "\"dictionary_training_repeated\":false,\"framing\":\"standard frames, default checksum off\",\"results\":[");
        let mut results = Vec::new();
        for level in levels {
            for (capacity, dictionary) in &dictionaries {
                fs::write(folder.join(format!("zstd-{capacity}.dict")), dictionary)
                    .map_err(ioerr)?;
                let start = Instant::now();
                let mut codec = Zstd::new(&api, dictionary, level)?;
                let preparation = start.elapsed().as_secs_f64();
                let name = format!("zstd{level}-{capacity}");
                eprintln!(
                    "Measuring {name}: {} rows, {} bytes",
                    rows.len() - 1,
                    raw.len()
                );
                let measured = benchmark(
                    &name,
                    &mut codec,
                    dictionary.len() as u64,
                    &raw,
                    &rows,
                    folder,
                )?;
                results.push(tagged(measured, &format!("\"representation\":\"zstd\",\"compression_level\":{level},\"requested_dictionary_bytes\":{capacity},\"per_record_checksum\":false,\"preparation_seconds\":{preparation:.6}")));
                fs::write(
                    folder.join("report.json"),
                    format!("{prefix}{}]}}\n", results.join(",")),
                )
                .map_err(ioerr)?;
            }
        }
        println!("{prefix}{}]}}", results.join(","));
        Ok(())
    }
    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.first().is_some_and(|arg| arg == "--zstd-train") {
            return train_zstd_dictionaries(&args);
        }
        if args.first().is_some_and(|arg| arg == "--zstd-from") {
            return zstd_frozen_experiments(&args);
        }
        if args.first().is_some_and(|arg| arg == "--eof-from") {
            return eof_experiments(&args);
        }
        if args.len() > 4 {
            return experiments(&args);
        }
        if args.len() != 4 {
            return Err("record-access TRAIN.psv TEST.psv NEW_DIRECTORY LIBZSTD.dll".into());
        }
        let ioerr = |e: std::io::Error| e.to_string();
        let folder = Path::new(&args[2]);
        fs::create_dir(folder).map_err(ioerr)?;
        let api = Api::load(Path::new(&args[3]))?;
        if unsafe { (api.version)() } != 10507 {
            return Err("benchmark requires libzstd 1.5.7".into());
        }
        eprintln!("Training BRe-Pair on training rows only: weighted PMI, minimum 32, four workers, no rule cap");
        let config = Config {
            buffer_bytes: 1_000_000_000,
            minimum_count: 32,
            score: scoring::Model::WeightedPmi,
            ..Default::default()
        };
        let start = Instant::now();
        let (model, report) =
            Frozen::train(&byte_model(), Path::new(&args[0]), &config, 4, None, |r| {
                eprintln!(
                    "round {}: {} rules, {} selected, {:.2}s",
                    r.index, r.total_rules, r.selected, r.seconds
                );
            })?;
        let training_seconds = start.elapsed().as_secs_f64();
        let stored = model.to_bytes()?;
        fs::write(folder.join("brepair.brc"), &stored).map_err(ioerr)?;
        let start = Instant::now();
        let mut brepair = Brepair::new(model)?;
        let prepare_seconds = start.elapsed().as_secs_f64();
        let raw = fs::read(&args[1]).map_err(ioerr)?;
        let rows = offsets(&raw);
        let protocol=format!(concat!("{{\"workload\":\"TPC-H-derived independent held-out rows; not SQL performance\",",
            "\"resident_random_access\":true,\"workers_during_measurement\":1,\"query_seed\":20261106,",
            "\"training_only_models\":true,\"zstd_version\":10507,\"brepair_minimum_count\":32,",
            "\"brepair_training_rounds\":{},\"brepair_training_seconds\":{:.6},\"brepair_preparation_seconds\":{:.6},",
            "\"brepair_record_header_bytes\":20,\"zstd_frames\":\"standard frames, default checksum off\",",
            "\"sequential_timing_includes_byte_comparison\":true,\"results\":["),report.rounds,training_seconds,prepare_seconds);
        let mut results = Vec::new();
        results.push(benchmark(
            "brepair",
            &mut brepair,
            stored.len() as u64,
            &raw,
            &rows,
            folder,
        )?);
        drop(brepair);
        let training = fs::read(&args[0]).map_err(ioerr)?;
        let train_rows = offsets(&training);
        let sizes: Vec<_> = train_rows.windows(2).map(|w| w[1] - w[0]).collect();
        let mut capacities = vec![0, 65536, 262144, stored.len()];
        capacities.sort_unstable();
        capacities.dedup();
        for capacity in capacities {
            let mut dict = vec![0; capacity];
            let start = Instant::now();
            if capacity > 0 {
                let n = api.check(unsafe {
                    (api.train)(
                        dict.as_mut_ptr().cast(),
                        dict.len(),
                        training.as_ptr().cast(),
                        sizes.as_ptr(),
                        u32::try_from(sizes.len()).map_err(|_| "too many training rows")?,
                    )
                })?;
                dict.truncate(n);
            }
            let dict_seconds = start.elapsed().as_secs_f64();
            fs::write(folder.join(format!("zstd-{capacity}.dict")), &dict).map_err(ioerr)?;
            let start = Instant::now();
            let mut zstd = Zstd::new(&api, &dict, 3)?;
            let setup_seconds = start.elapsed().as_secs_f64();
            eprintln!("zstd {capacity}: training {dict_seconds:.3}s; setup {setup_seconds:.6}s");
            let name = format!("zstd3-{capacity}");
            let mut measured = benchmark(&name, &mut zstd, dict.len() as u64, &raw, &rows, folder)?;
            measured.pop();
            measured.push_str(&format!(",\"requested_dictionary_bytes\":{capacity},\"dictionary_training_seconds\":{dict_seconds:.6},\"preparation_seconds\":{setup_seconds:.6},\"matches_brepair_model_budget\":{}}}", capacity == stored.len()));
            results.push(measured);
            let json = format!("{protocol}{}]}}\n", results.join(","));
            fs::write(folder.join("report.json"), &json).map_err(ioerr)?;
        }
        println!("{protocol}{}]}}", results.join(","));
        Ok(())
    }
}
fn main() {
    #[cfg(windows)]
    if let Err(error) = windows::run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    #[cfg(not(windows))]
    {
        eprintln!("This research benchmark currently requires Windows/libzstd.dll");
        std::process::exit(1);
    }
}
