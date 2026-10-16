import { anomalyTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real scikit-learn elliptic envelope, labelled. Requires ML_TEST_PYTHON.
describeTaskContract(anomalyTarget("robust_covariance", true));
