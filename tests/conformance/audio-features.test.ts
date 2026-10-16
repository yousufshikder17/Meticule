import { audioTarget } from "./audio-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real audio runner (MFCC summary embeddings). Requires ML_TEST_AUDIO_PYTHON; CI's main test job provides it.
describeTaskContract(audioTarget("audio_features", false, ["mean_pairwise_cosine"]));
