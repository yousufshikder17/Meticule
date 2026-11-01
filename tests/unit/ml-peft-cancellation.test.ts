import {execFile} from "node:child_process";
import {mkdtemp,readFile,rm,stat,writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {basename,dirname,join,resolve} from "node:path";
import {promisify} from "node:util";
import {expect,it} from "vitest";
import {callRunner} from "../../src/ml/runner-process.js";

const python=process.env.ML_TEST_PEFT_PYTHON;
it.runIf(Boolean(python))("cancellation exits the actual guarded interpreter, including a Windows venv launcher",async()=>{
  const root=await mkdtemp(join(tmpdir(),"meticule-public-cancel-")),marker=join(root,"pid.json"),script=join(root,"guarded.py");
  const controller=new AbortController();
  let ownedTemporary:string|undefined;
  try{
    await writeFile(script,["import sys,os,json,time,tempfile",`sys.path.insert(0,${JSON.stringify(resolve("ml"))})`,"import runner_parent_guard","runner_parent_guard.start_parent_guard()","temporary=tempfile.mkdtemp(prefix='meticule-public-owned-temp-')","open(os.path.join(temporary,'asset'),'w').write('controlled fixture')",`open(${JSON.stringify(marker)},'w').write(json.dumps({'pid':os.getpid(),'temporary':temporary}))`,"time.sleep(120)"].join("\n"));
    const outcome=callRunner({label:"guard test",python:python!,script,timeoutMs:150_000,signal:controller.signal,privateTemporaryDirectory:true,request:{command:"guard_probe"}}).then(()=>null,error=>error);
    let pid:number|undefined;const deadline=Date.now()+30_000;
    while(!pid&&Date.now()<deadline){const state=await readFile(marker,"utf8").then(JSON.parse).catch(()=>undefined);pid=state?.pid;ownedTemporary=state?.temporary;if(!pid)await new Promise(resolve=>setTimeout(resolve,50));}
    expect(pid,"the guarded interpreter must actually start").toBeGreaterThan(0);controller.abort();
    expect((await outcome)?.category).toBe("cancelled");
    const {stdout}=await promisify(execFile)(python!,["-I","-c","import psutil,sys;print(int(psutil.pid_exists(int(sys.argv[1]))))",String(pid)],{timeout:30_000});
    expect(stdout.trim()).toBe("0");
    await expect(stat(ownedTemporary!)).rejects.toMatchObject({code:"ENOENT"});
  }finally{
    controller.abort();await rm(root,{recursive:true,force:true});
    if(ownedTemporary&&resolve(dirname(ownedTemporary))===resolve(tmpdir())&&basename(ownedTemporary).startsWith("meticule-public-owned-temp-"))await rm(ownedTemporary,{recursive:true,force:true});
  }
},90_000);
