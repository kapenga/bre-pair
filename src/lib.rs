#[cfg(not(target_pointer_width = "64"))]
compile_error!("bre-pair requires a 64-bit target for large buffers, indexes, and file sizes");

pub mod api;
pub mod archive;
pub mod archive_io;
pub mod atomic_output;
#[cfg(feature = "audit")]
mod audit;
pub mod batch;
mod batch_experiments;
#[cfg(feature = "bytecode-experiments")]
pub mod bytecode;
pub mod codec;
pub mod columns;
pub mod context_entropy;
pub mod disk_seed;
pub mod document_support;
pub mod engine;
pub mod entropy_engine;
mod entropy_registry;
pub mod field_engine;
pub mod grammar;
pub mod grammar_reparse;
pub mod huffman;
pub mod huffman_archive;
pub mod huffman_dictionary;
pub mod huffman_pipeline;
pub mod joint;
pub mod language_prior;
pub mod lookahead;
pub mod maximal;
pub mod memory;
pub mod packed;
pub mod pair_map;
pub mod pathpiece;
pub mod prefix_beam;
#[cfg(feature = "profile")]
pub mod profile;
pub mod prune;
pub mod record_frame;
pub mod record_refit;
pub mod registry;
pub mod repeated;
pub mod scalar_bonds;
pub mod scoring;
pub mod search;
pub mod sequence_archive;
pub mod shake;
pub mod shared_huffman;
mod symbol_counts;
pub mod targeted;
pub mod token_stream;
pub mod tokenizer;
pub mod training_progress;
mod utf8_prior;
pub mod vocabulary;
