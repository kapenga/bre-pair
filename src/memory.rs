//! Allocation accounting used by the public executable. Research binaries can
//! keep their normal allocator. Reallocations reserve both old and new storage.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static USED: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LIMIT: AtomicUsize = AtomicUsize::new(0);

pub struct Allocator;

fn reserve(bytes: usize) {
    let result = USED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
        old.checked_add(bytes).filter(|&next| {
            LIMIT.load(Ordering::Relaxed) == 0 || next <= LIMIT.load(Ordering::Relaxed)
        })
    });
    match result {
        Ok(old) => {
            PEAK.fetch_max(old + bytes, Ordering::Relaxed);
        }
        Err(_) => exhausted(),
    }
}

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        reserve(layout.size());
        let ptr = System.alloc(layout);
        if ptr.is_null() {
            USED.fetch_sub(layout.size(), Ordering::Relaxed);
            exhausted();
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        reserve(layout.size());
        let ptr = System.alloc_zeroed(layout);
        if ptr.is_null() {
            USED.fetch_sub(layout.size(), Ordering::Relaxed);
            exhausted();
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        USED.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        reserve(size);
        let result = System.realloc(ptr, layout, size);
        if result.is_null() {
            USED.fetch_sub(size, Ordering::Relaxed);
            exhausted();
        }
        USED.fetch_sub(layout.size(), Ordering::Relaxed);
        result
    }
}

fn exhausted() -> ! {
    const MESSAGE:&[u8]=b"brepair: memory limit exceeded; increase --memory or reduce the input/candidate budget.\n";
    #[cfg(windows)]
    unsafe {
        let mut written = 0;
        win::WriteFile(
            win::GetStdHandle(-12i32 as u32),
            MESSAGE.as_ptr(),
            MESSAGE.len() as u32,
            &mut written,
            std::ptr::null_mut(),
        );
        win::ExitProcess(2)
    }
    #[cfg(not(windows))]
    {
        // std's stderr path uses no formatted heap buffer for these literals.
        use std::io::Write;
        let _ = std::io::stderr().write_all(MESSAGE);
        std::process::exit(2)
    }
}

pub fn heap_bytes() -> usize {
    USED.load(Ordering::Relaxed)
}
pub fn peak_heap_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}
pub fn heap_limit() -> usize {
    LIMIT.load(Ordering::Relaxed)
}

pub struct Limit {
    pub total: usize,
    pub heap: usize,
    #[cfg(windows)]
    handle: *mut std::ffi::c_void,
}

impl Limit {
    /// On Windows the job caps private commit as a second backstop. Shared
    /// executable/DLL pages have separate headroom. No worker-process spawning.
    pub fn set(total: usize, workers: usize) -> Result<Self, String> {
        let reserve = 32 * 1024 * 1024usize + workers * 256 * 1024;
        let heap = total
            .checked_sub(reserve)
            .filter(|&n| n >= 16 * 1024 * 1024)
            .ok_or("memory limit is too small; use at least 64MB")?;
        if heap_bytes() > heap {
            return Err("memory limit is below current allocations".into());
        }
        #[cfg(windows)]
        let handle = unsafe {
            let handle = win::CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
            if handle.is_null() {
                return Err(std::io::Error::last_os_error().to_string());
            }
            let mut info = win::JobLimits::default();
            info.basic.flags = 0x100;
            info.process_memory = total - 16 * 1024 * 1024;
            if win::SetInformationJobObject(
                handle,
                9,
                &info as *const _ as *const _,
                std::mem::size_of_val(&info) as u32,
            ) == 0
                || win::AssignProcessToJobObject(handle, win::GetCurrentProcess()) == 0
            {
                let error = std::io::Error::last_os_error();
                win::CloseHandle(handle);
                return Err(format!("cannot install process memory limit: {error}"));
            }
            handle
        };
        LIMIT.store(heap, Ordering::Relaxed);
        Ok(Self {
            total,
            heap,
            #[cfg(windows)]
            handle,
        })
    }
}
impl Drop for Limit {
    fn drop(&mut self) {
        LIMIT.store(0, Ordering::Relaxed);
        #[cfg(windows)]
        unsafe {
            win::CloseHandle(self.handle);
        }
    }
}

#[derive(Default, Debug)]
pub struct Process {
    pub peak_resident: usize,
    pub peak_private: usize,
}
pub fn process() -> Process {
    #[cfg(windows)]
    unsafe {
        let mut info = win::Counters {
            size: std::mem::size_of::<win::Counters>() as u32,
            ..Default::default()
        };
        if win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut info, info.size) != 0 {
            return Process {
                peak_resident: info.peak_working,
                peak_private: info.peak_commit,
            };
        }
    }
    Process::default()
}

pub fn parse_bytes(value: &str) -> Result<usize, String> {
    let split = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let n = value[..split]
        .parse::<usize>()
        .map_err(|_| "invalid memory size")?;
    let factor = match value[split..].to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kb" => 1000,
        "mb" => 1000000,
        "gb" => 1000000000,
        "kib" => 1024,
        "mib" => 1024 * 1024,
        "gib" => 1024 * 1024 * 1024,
        _ => return Err("use bytes, KB/MB/GB, or KiB/MiB/GiB".into()),
    };
    n.checked_mul(factor).ok_or("memory size overflow".into())
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    pub struct Basic {
        pub process_time: i64,
        pub job_time: i64,
        pub flags: u32,
        pub min_working: usize,
        pub max_working: usize,
        pub active: u32,
        pub affinity: usize,
        pub priority: u32,
        pub scheduling: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct JobLimits {
        pub basic: Basic,
        pub io: [u64; 6],
        pub process_memory: usize,
        pub job_memory: usize,
        pub peak_process: usize,
        pub peak_job: usize,
    }
    #[repr(C)]
    #[derive(Default)]
    pub struct Counters {
        pub size: u32,
        pub faults: u32,
        pub peak_working: usize,
        pub working: usize,
        pub peak_paged: usize,
        pub paged: usize,
        pub peak_nonpaged: usize,
        pub nonpaged: usize,
        pub commit: usize,
        pub peak_commit: usize,
        pub private: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> *mut c_void;
        pub fn SetInformationJobObject(
            job: *mut c_void,
            class: u32,
            info: *const c_void,
            length: u32,
        ) -> i32;
        pub fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn CloseHandle(handle: *mut c_void) -> i32;
        pub fn GetStdHandle(which: u32) -> *mut c_void;
        pub fn WriteFile(
            handle: *mut c_void,
            bytes: *const u8,
            length: u32,
            written: *mut u32,
            overlap: *mut c_void,
        ) -> i32;
        pub fn ExitProcess(code: u32) -> !;
        pub fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            info: *mut Counters,
            length: u32,
        ) -> i32;
    }
}
