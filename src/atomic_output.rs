//! Publish completed files without overwriting an existing destination. A hard
//! memory-limit exit can leave a .partial file, never a final-looking result.
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct Output {
    destination: PathBuf,
    temporary: PathBuf,
    committed: bool,
}
impl Output {
    pub fn new(destination: &Path) -> Result<Self, String> {
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let destination = fs::canonicalize(parent)
            .map_err(|e| format!("output directory: {e}"))?
            .join(
                destination
                    .file_name()
                    .ok_or("output path needs a file name")?,
            );
        if destination.exists() {
            return Err(format!("output exists: {}", destination.display()));
        }
        let mut name = std::ffi::OsString::from(".");
        name.push(
            destination
                .file_name()
                .ok_or("output path needs a file name")?,
        );
        name.push(format!(
            ".{}.{}.partial",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary = destination.with_file_name(name);
        if temporary.exists() {
            return Err("temporary output already exists".into());
        }
        Ok(Self {
            destination,
            temporary,
            committed: false,
        })
    }
    pub fn path(&self) -> &Path {
        &self.temporary
    }
    pub fn destination(&self) -> &Path {
        &self.destination
    }
    pub fn commit(mut self) -> Result<(), String> {
        // A hard link is atomic and fails if another process created OUTPUT.
        // rename() would overwrite on Unix, so it is deliberately not used.
        fs::hard_link(&self.temporary, &self.destination)
            .map_err(|e| format!("cannot publish {}: {e}", self.destination.display()))?;
        fs::remove_file(&self.temporary).map_err(|e| e.to_string())?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.temporary);
        }
    }
}
