use bre_pair::engine::Compressed;
use bre_pair::grammar::Grammar;
use bre_pair::maximal::NativeCompressed;
use bre_pair::{archive, sequence_archive};

#[test]
fn archive_headers_preserve_terabyte_sizes_and_large_field_counts() {
    // Framing-only fixtures; no claim of a valid decoded corpus or huge RAM
    // allocation. Both serializers must preserve their 64-bit metadata fields.
    let input_bytes = 1_000_000_000_000;
    let field_count = (1u64 << 32) + 7;
    let binary = Compressed {
        grammar: Grammar::default(),
        blocks: vec![],
        input_bytes,
        field_count,
    };
    let restored = archive::decode(&archive::encode(&binary).unwrap()).unwrap();
    assert_eq!(restored.input_bytes, input_bytes);
    assert_eq!(restored.field_count, field_count);
    let native = NativeCompressed {
        rules: vec![],
        blocks: vec![],
        input_bytes,
        field_count,
    };
    let restored = sequence_archive::decode(&sequence_archive::encode(&native).unwrap()).unwrap();
    assert_eq!(restored.input_bytes, input_bytes);
    assert_eq!(restored.field_count, field_count);
}

#[test]
fn a_large_block_length_cannot_wrap_into_a_small_valid_payload() {
    for magic in [b"BRPAIR01", b"BRPNAR01"] {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(magic);
        bytes.extend_from_slice(&0u64.to_le_bytes()); // input bytes
        bytes.extend_from_slice(&1u64.to_le_bytes()); // fields
        bytes.extend_from_slice(&0u32.to_le_bytes()); // rules
        bytes.extend_from_slice(&1u32.to_le_bytes()); // blocks
        bytes.extend_from_slice(&0u64.to_le_bytes()); // dictionary bits
        bytes.push(9);
        bytes.extend_from_slice(&((1u64 << 32) + 1).to_le_bytes());
        bytes.extend_from_slice(&[0, 1]); // valid EOF payload only if length wraps to 1
        let error = if magic == b"BRPAIR01" {
            archive::decode(&bytes).err()
        } else {
            sequence_archive::decode(&bytes).err()
        };
        assert_eq!(error.as_deref(), Some("truncated archive"));
    }
}
