import { describeTaskContract } from "./task-suite.js";
import { transformersTarget } from "./transformers-fixtures.js";

// Real transformers runner over a tiny offline fixture. Requires ML_TEST_TRANSFORMERS_PYTHON; the transformers CI job provides it.
describeTaskContract(transformersTarget("image_embedding_classifier", "tiny-vit-encoder", "image", true, ["accuracy"]));
