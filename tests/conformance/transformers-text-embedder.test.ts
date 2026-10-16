import { describeTaskContract } from "./task-suite.js";
import { transformersTarget } from "./transformers-fixtures.js";

// Real transformers runner over a tiny offline fixture. Requires ML_TEST_TRANSFORMERS_PYTHON; the transformers CI job provides it.
describeTaskContract(transformersTarget("text_embedder", "tiny-bert-encoder", "text", true, ["retrieval_precision_at_1", "retrieval_mrr"]));
