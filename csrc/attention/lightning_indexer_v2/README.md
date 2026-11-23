# A6 BF16 LightningIndexerV2 with CANN 9.3

The BF16 binding calls CANN's `aclnnLightningIndexerV2`. Its kernel is
delivered by the CANN operator package, separately from the vllm-ascend wheel.

On Ascend960DT a vector core can have decode reductions assigned even when
its paired cube has no main work. The CANN 9.3 kernel skips these reductions.
For a four-token query with 130 KV tokens, this leaves the last two output
rows unwritten. The patch reads the vector assignment independently.

Apply the patch in an [ops-transformer checkout](https://github.com/hicann/ops-transformer/tree/75b6beb460eb32644a6c87fa5f9d76b756d17b4d)
matching the CANN 9.3 sources. From that checkout, with the CANN build
environment active, run

```sh
patch_file="$VLLM_ASCEND_SRC/csrc/attention/lightning_indexer_v2/cann-9.3-reduction.patch"
git apply --check "$patch_file"
git apply "$patch_file"
bash build.sh --pkg --ops=lightning_indexer_v2 --soc=ascend960dt
```

`VLLM_ASCEND_SRC` is the absolute path to this vllm-ascend checkout. Install
and activate the resulting CANN operator package for the worker process.
Then run the following regression from the vllm-ascend checkout on an A6.
The four rows must cover 127, 128, 129 and 130 distinct KV positions.

```sh
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_lightning_indexer_v2.py -k four_query_rows
```
