//! Private ABI v1 for the Python package. All allocations are freed by Rust.
use bre_pair::{api::ByteTokenizer, tokenizer::ParseScratch};
use std::{
    cell::RefCell,
    ffi::{c_char, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    sync::{Arc, Mutex},
};

thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("").unwrap());}
pub struct Handle {
    model: Arc<ByteTokenizer>,
    scratch: Mutex<ParseScratch>,
}
#[repr(C)]
pub struct Bytes {
    pub data: *mut u8,
    pub length: usize,
}
#[repr(C)]
pub struct Ids {
    pub data: *mut u32,
    pub length: usize,
}

fn guard<T>(fallback: T, run: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok(value)) => value,
        error => {
            let message = match error {
                Ok(Err(message)) => message,
                _ => "native operation panicked".into(),
            };
            ERROR.with(|slot| {
                *slot.borrow_mut() = CString::new(message.replace('\0', "\\0")).unwrap()
            });
            fallback
        }
    }
}
unsafe fn input<'a, T>(data: *const T, length: usize) -> Result<&'a [T], String> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null()
        || length > isize::MAX as usize / std::mem::size_of::<T>()
        || (data as usize) % std::mem::align_of::<T>() != 0
    {
        return Err("invalid native input bounds".into());
    }
    Ok(std::slice::from_raw_parts(data, length))
}
fn bytes(value: Vec<u8>) -> Bytes {
    let mut value = value.into_boxed_slice();
    let result = Bytes {
        data: value.as_mut_ptr(),
        length: value.len(),
    };
    std::mem::forget(value);
    result
}
fn ids(value: Vec<u32>) -> Ids {
    let mut value = value.into_boxed_slice();
    let result = Ids {
        data: value.as_mut_ptr(),
        length: value.len(),
    };
    std::mem::forget(value);
    result
}

#[no_mangle]
pub extern "C" fn brepair_abi_version() -> u32 {
    1
}
#[no_mangle]
pub extern "C" fn brepair_last_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}
/// # Safety
/// `path` must address `length` readable bytes for this call, or be null when
/// length is zero. Free the returned handle exactly once with this library.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_load(path: *const u8, length: usize) -> *mut Handle {
    guard(std::ptr::null_mut(), || {
        let path =
            std::str::from_utf8(input(path, length)?).map_err(|_| "model path must be UTF-8")?;
        Ok(Box::into_raw(Box::new(Handle {
            model: Arc::new(ByteTokenizer::load(Path::new(path))?),
            scratch: Mutex::new(ParseScratch::default()),
        })))
    })
}
/// # Safety
/// `data` must address `length` readable bytes for this call, or be null when
/// length is zero. Free the returned handle exactly once with this library.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_from_bytes(
    data: *const u8,
    length: usize,
) -> *mut Handle {
    guard(std::ptr::null_mut(), || {
        Ok(Box::into_raw(Box::new(Handle {
            model: Arc::new(ByteTokenizer::from_bytes(input(data, length)?)?),
            scratch: Mutex::new(ParseScratch::default()),
        })))
    })
}
/// # Safety
/// A nonnull handle must be live and must not be freed during this call.
/// The returned handle has independent ownership and needs its own free call.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_fork(handle: *const Handle) -> *mut Handle {
    guard(std::ptr::null_mut(), || {
        let handle = handle.as_ref().ok_or("closed tokenizer")?;
        Ok(Box::into_raw(Box::new(Handle {
            model: Arc::clone(&handle.model),
            scratch: Mutex::new(ParseScratch::default()),
        })))
    })
}
/// # Safety
/// A nonnull handle must be an owned pointer returned by this library, not
/// previously freed. No other call may use it concurrently or afterward.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_free(handle: *mut Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
/// # Safety
/// A nonnull handle must remain live for this call without concurrent freeing.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_size(handle: *const Handle) -> usize {
    guard(0, || {
        Ok(handle
            .as_ref()
            .ok_or("closed tokenizer")?
            .model
            .vocabulary_size())
    })
}
/// # Safety
/// The handle must remain live; nonnull output must be aligned, writable and
/// exclusively borrowed. On success free its new allocation with bytes_free.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_bytes(handle: *const Handle, output: *mut Bytes) -> i32 {
    guard(-1, || {
        let out = output.as_mut().ok_or("missing byte output")?;
        *out = bytes(handle.as_ref().ok_or("closed tokenizer")?.model.to_bytes());
        Ok(0)
    })
}
/// # Safety
/// The handle must remain live. Input must address `length` readable bytes
/// (null is valid for zero length). Output must be aligned, writable and
/// exclusively borrowed; on success release its allocation with ids_free.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_encode(
    handle: *const Handle,
    data: *const u8,
    length: usize,
    output: *mut Ids,
) -> i32 {
    guard(-1, || {
        let out = output.as_mut().ok_or("missing ID output")?;
        let handle = handle.as_ref().ok_or("closed tokenizer")?;
        let mut scratch = handle
            .scratch
            .lock()
            .map_err(|_| "tokenizer workspace is poisoned")?;
        *out = ids(handle
            .model
            .encode(input(data, length)?, &mut scratch)
            .to_vec());
        Ok(0)
    })
}
/// # Safety
/// The handle must remain live. Input must address `length` aligned, readable
/// u32 IDs (null is valid for zero length). Output must be aligned, writable
/// and exclusive; on success release its allocation with bytes_free.
#[no_mangle]
pub unsafe extern "C" fn brepair_tokenizer_decode(
    handle: *const Handle,
    data: *const u32,
    length: usize,
    maximum: usize,
    output: *mut Bytes,
) -> i32 {
    guard(-1, || {
        let out = output.as_mut().ok_or("missing byte output")?;
        *out = bytes(
            handle
                .as_ref()
                .ok_or("closed tokenizer")?
                .model
                .decode(input(data, length)?, maximum)?,
        );
        Ok(0)
    })
}
/// # Safety
/// A nonnull value must be the unmodified, still-owned allocation descriptor
/// returned by this library. Free it once, with no outstanding accesses.
#[no_mangle]
pub unsafe extern "C" fn brepair_bytes_free(value: Bytes) {
    if !value.data.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            value.data,
            value.length,
        )));
    }
}
/// # Safety
/// A nonnull value must be the unmodified, still-owned allocation descriptor
/// returned by this library. Free it once, with no outstanding accesses.
#[no_mangle]
pub unsafe extern "C" fn brepair_ids_free(value: Ids) {
    if !value.data.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            value.data,
            value.length,
        )));
    }
}
