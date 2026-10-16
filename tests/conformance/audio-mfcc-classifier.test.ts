import { audioTarget } from "./audio-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real audio runner (softmax head over MFCC summaries). Requires ML_TEST_AUDIO_PYTHON; CI's main test job provides it.
describeTaskContract(audioTarget("audio_mfcc_classifier", true, ["accuracy", "log_loss"], { l2: 0.1 }));
