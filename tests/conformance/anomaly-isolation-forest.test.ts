import { anomalyTarget } from "./series-fixtures.js";
import { describeTaskContract } from "./task-suite.js";

// Real scikit-learn isolation forest, labelled. Requires ML_TEST_PYTHON.
describeTaskContract(anomalyTarget("isolation_forest", true));
