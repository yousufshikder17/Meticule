import {execFile} from "node:child_process";
import {resolve} from "node:path";
import {promisify} from "node:util";
import {expect,it} from "vitest";

const python=process.env.ML_TEST_PEFT_PYTHON;
for(const [name,count]of [["check_lora.py",27],["check_lora_backend.py",28],["check_lora_audit.py",20],["check_provisioned_lora.py",23]]as const){
  it.runIf(Boolean(python))(`real public adapter computation: ${name}`,async()=>{
    const {stdout}=await promisify(execFile)(python!,["-B",resolve("tests/python",name)],{timeout:600_000,maxBuffer:16_000_000,windowsHide:true});
    const checks=JSON.parse(stdout.trim().split("\n").at(-1)!).results as Record<string,boolean>;
    expect(Object.keys(checks)).toHaveLength(count);expect(Object.entries(checks).filter(([,passed])=>passed!==true)).toEqual([]);
  },650_000);
}
it.runIf(process.env.ML_PEFT_JOB==="true")("CI supplies the approved local adapter interpreter",()=>{
  expect(python).toBeTruthy();
});
