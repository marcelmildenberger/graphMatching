from graphMatching.gma_pipeline import (
    prepare_similarity_graph_inputs,
    run_embedding_and_matching,
    validate_gma_configuration,
)


def run_gma(GLOBAL_CONFIG, ENC_CONFIG, EMB_CONFIG, ALIGN_CONFIG, eve_enc_hash, alice_enc_hash, eve_emb_hash, alice_emb_hash):
    """Run the graph matching attack pipeline for the current experiment."""
    validate_gma_configuration(GLOBAL_CONFIG, ENC_CONFIG, ALIGN_CONFIG)
    similarity_state = prepare_similarity_graph_inputs(
        GLOBAL_CONFIG,
        ENC_CONFIG,
        EMB_CONFIG,
        eve_enc_hash,
        alice_enc_hash,
    )
    return run_embedding_and_matching(
        GLOBAL_CONFIG,
        ENC_CONFIG,
        EMB_CONFIG,
        ALIGN_CONFIG,
        eve_enc_hash,
        alice_enc_hash,
        eve_emb_hash,
        alice_emb_hash,
        similarity_state,
    )
