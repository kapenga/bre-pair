//! Opt-in coarse timers and registry counters. Absent from ordinary builds.
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Default)]
struct Measurement {
    calls: u64,
    seconds: f64,
    thread_cpu_seconds: f64,
    amount: u64,
}

fn measurements() -> &'static Mutex<BTreeMap<&'static str, Measurement>> {
    static DATA: OnceLock<Mutex<BTreeMap<&'static str, Measurement>>> = OnceLock::new();
    DATA.get_or_init(Mutex::default)
}

fn record(name: &'static str, calls: u64, seconds: f64, cpu: f64, amount: u64) {
    let mut data = measurements()
        .lock()
        .expect("poisoned profile measurements");
    let entry = data.entry(name).or_default();
    entry.calls += calls;
    entry.seconds += seconds;
    entry.thread_cpu_seconds += cpu;
    entry.amount += amount;
}

pub(crate) fn counter(name: &'static str, amount: u64) {
    record(name, 0, 0.0, 0.0, amount);
}

/// Merge local phase totals once, without a shared timer/lock per field.
pub(crate) fn elapsed(name: &'static str, calls: u64, seconds: f64) {
    record(name, calls, seconds, 0.0, 0);
}

pub struct Span {
    name: &'static str,
    started: Instant,
    cpu: Option<f64>,
}

impl Span {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            started: Instant::now(),
            cpu: thread_cpu_seconds(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let seconds = self.started.elapsed().as_secs_f64();
        let cpu = self
            .cpu
            .zip(thread_cpu_seconds())
            .map_or(0.0, |(before, after)| after - before);
        record(self.name, 1, seconds, cpu, 0);
    }
}

/// One writer's local metrics, merged only once when its worker finishes.
#[derive(Default)]
pub(crate) struct RegistryMetrics {
    pub deltas: u64,
    pub flushes: u64,
    pub updates: u64,
    pub probes: u64,
    pub registrations: u64,
    pub registration_collisions: u64,
    pub counter_retries: u64,
    pub locks: u64,
    pub lock_seconds: f64,
    pub sort_seconds: f64,
    pub apply_seconds: f64,
    pub grow_seconds: f64,
    pub growth_requests: u64,
    pub endpoint_checks: u64,
    pub skipped_pair_deltas: u64,
    pub start_counter_updates: u64,
    pub overlap_counter_updates: u64,
}

impl RegistryMetrics {
    pub fn record(&self) {
        for (name, amount) in [
            ("registry.deltas", self.deltas),
            ("registry.flushes", self.flushes),
            ("registry.updates", self.updates),
            ("registry.probes", self.probes),
            ("registry.registrations", self.registrations),
            (
                "registry.registration_collisions",
                self.registration_collisions,
            ),
            ("registry.counter_retries", self.counter_retries),
            ("cutoff.endpoint_checks", self.endpoint_checks),
            ("cutoff.skipped_pair_deltas", self.skipped_pair_deltas),
            ("cutoff.start_counter_updates", self.start_counter_updates),
            (
                "cutoff.overlap_counter_updates",
                self.overlap_counter_updates,
            ),
        ] {
            record(name, 0, 0.0, 0.0, amount);
        }
        for (name, calls, seconds) in [
            ("registry.sort", self.flushes, self.sort_seconds),
            ("registry.apply", self.flushes, self.apply_seconds),
            ("registry.shard_lock", self.locks, self.lock_seconds),
            ("registry.grow", self.growth_requests, self.grow_seconds),
        ] {
            record(name, calls, seconds, 0.0, 0);
        }
    }
}

/// CSV timers are inclusive. Worker seconds sum across threads; they must not
/// be added to coordinator wall time. Registry apply includes lock/grow time.
pub fn write(path: &std::path::Path) -> Result<(), String> {
    let data = measurements().lock().map_err(|e| e.to_string())?;
    let mut file = std::fs::File::create_new(path).map_err(|e| e.to_string())?;
    writeln!(file, "name,calls,seconds,thread_cpu_seconds,amount").map_err(|e| e.to_string())?;
    for (name, value) in data.iter() {
        writeln!(
            file,
            "{name},{},{:.9},{:.9},{}",
            value.calls, value.seconds, value.thread_cpu_seconds, value.amount
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[cfg(windows)]
pub fn thread_cpu_seconds() -> Option<f64> {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetThreadTimes(
            thread: *mut std::ffi::c_void,
            created: *mut FileTime,
            exited: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let (mut created, mut exited, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // The pseudo-handle is valid for this thread; all four output pointers
    // refer to correctly sized, writable FILETIME-compatible values.
    let success = unsafe {
        GetThreadTimes(
            GetCurrentThread(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
    };
    (success != 0).then(|| {
        let ticks = |time: FileTime| ((time.high as u64) << 32) | time.low as u64;
        (ticks(kernel) + ticks(user)) as f64 * 1e-7
    })
}

#[cfg(not(windows))]
pub fn thread_cpu_seconds() -> Option<f64> {
    None
}

/// Pin only the calling benchmark thread to the first processor allowed for
/// this process. Other applications and system settings are unchanged.
#[cfg(windows)]
pub fn pin_benchmark_thread() -> Result<Option<u32>, String> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetProcessAffinityMask(
            process: *mut std::ffi::c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
    }
    let (mut allowed, mut system) = (0usize, 0usize);
    // Pseudo handles refer to this process/thread; output masks have native
    // pointer width. The nonzero mask comes from the OS's allowed processors.
    let result = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut allowed, &mut system) };
    if result == 0 || allowed == 0 {
        return Err("cannot query benchmark CPU affinity".into());
    }
    let cpu = allowed.trailing_zeros();
    if unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) } == 0 {
        return Err("cannot pin benchmark thread".into());
    }
    Ok(Some(cpu))
}
#[cfg(not(windows))]
pub fn pin_benchmark_thread() -> Result<Option<u32>, String> {
    Ok(None)
}
