import { describeTaskContract } from "./task-suite.js";
import { transformersTarget } from "./transformers-fixtures.js";

// Real transformers runner over a tiny offline wav2vec2 fixture. Requires ML_TEST_TRANSFORMERS_PYTHON; the transformers CI job provides it.
describeTaskContract(transformersTarget("audio_classifier", "tiny-wav2vec2-tones", "audio", true, ["accuracy", "log_loss"]));
