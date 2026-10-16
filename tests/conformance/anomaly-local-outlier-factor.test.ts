import { anomalyTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real scikit-learn local outlier factor, unlabelled. Requires ML_TEST_PYTHON.
describeTaskContract(anomalyTarget("local_outlier_factor", false));
