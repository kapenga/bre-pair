//! Compatibility entry point; implementation is shared with the Rust library.
#[global_allocator]
static ALLOCATOR: bre_pair::memory::Allocator = bre_pair::memory::Allocator;
fn main() {
    let _limit = match bre_pair::memory::Limit::set(8_000_000_000, 4) {
        Ok(limit) => limit,
        Err(error) => {
            eprintln!("column-pilot: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) =
        bre_pair::columns::run_research_cli(&std::env::args().skip(1).collect::<Vec<_>>())
    {
        eprintln!("column-pilot: {error}");
        std::process::exit(1);
    }
}
