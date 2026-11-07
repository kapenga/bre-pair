//! Research CLI; the same implementation is also used by brepair.exe.
#[path = "../cli/tokenizer.rs"]
mod tokenizer;
fn main() -> std::process::ExitCode {
    match tokenizer::run(std::env::args().skip(1).collect(), true) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
