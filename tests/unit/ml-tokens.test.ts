import {randomUUID} from "node:crypto";
import {expect,it} from "vitest";
import {DatasetSchema,validateRows} from "../../src/ml/domain.js";
import {defaultMetrics} from "../../src/ml/metrics.js";

it("accepts bounded integer token IDs and refuses ambiguous or executable cells",()=>{
  const schema=DatasetSchema.parse({id:randomUUID(),columns:[{name:"seq",type:"tokens"}]});
  for(const seq of ["0","1 2 3",Array(512).fill("999999").join(" ")])expect(validateRows(schema,[{seq}])).toEqual([{seq}]);
  for(const seq of ["","01","-1","1.2","1  2","1\n","1\r"," 1","1000000",Array(513).fill("1").join(" "),"__import__('os')"])
    expect(()=>validateRows(schema,[{seq}])).toThrow();
  expect(defaultMetrics("language_modeling",false)).toEqual(["token_cross_entropy","perplexity"]);
});
