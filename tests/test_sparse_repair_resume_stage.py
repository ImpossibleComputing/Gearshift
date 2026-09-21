import importlib.util,json,os,pathlib,tempfile,time,unittest
spec=importlib.util.spec_from_file_location('resume_stage',str(pathlib.Path(__file__).resolve().parents[1]/'scripts/sparse_repair_resume_stage.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return p

def fixture(root):
    tid='task/1';hist={'task_id':tid,'prompt_ids':[1],'prefix_ids':[1,2],'bridge_ids':[151668]}
    hp=put(root/'saved/history.json',hist);task={'task_id':tid,'history_path':'saved/history.json','history_sha256':m.sha(hp)}
    d={'mapper':{'sha256':'mapper'},'seeds':{tid:[{'seed_index':i,'stream':s,'answer_seed':100+i}for i,s in enumerate(('answer_small','coverage_answer_1','coverage_answer_2'))]}}
    folder=root/m.RESULT/'screen/tasks/task__1'
    tpl={'task_id':tid,'prompt_ids':[1,151668],'prefix_ids':[1],'bridge_ids':[151668],
         'declaration_sha256':m.DECL_SHA,'enable_thinking':False,'rendered_template':'<think>\n\n</think>'}
    tp=put(folder/'prompt_only_template.json',tpl);files=[]
    for condition in m.CONDITIONS:
      for seed in d['seeds'][tid]:
        dest=folder/condition/f"seed_{seed['seed_index']}"
        outer={'experiment_id':'sparse_repair_01','cohort':'development_screen','task_id':tid,
         'condition':condition,**seed,'declaration_sha256':m.DECL_SHA,
         'history_sha256':m.sha(tp) if condition=='P' else task['history_sha256'],
         'original_source_history_sha256':task['history_sha256'],'mapper_sha256':'mapper',
         'implementation_sha256':m.IMPLEMENTATION_SHA,'teacher_answer_prefix_supplied':False}
        h={k:tpl[k]for k in ('task_id','prompt_ids','prefix_ids','bridge_ids')}if condition=='P' else hist
        ident={'schema':1,'sampler':'answer','task_id':tid,'stream':seed['stream'],'seed':seed['answer_seed'],'cap':4096,
         'model':{'model_id':'Qwen/Qwen3-8B','revision':'b968826d9c46dd6066d109eabc6255188de91218',
           'runtime_identity_sha256':m.RUNTIME_SHA,'declaration_sha256':m.DECL_SHA,
           'cache_identity_sha256':m.digest({**outer,'conditioning':m.digest(h)})},
         'conditioning':{'history_sha256':m.digest(h),'prefix_ids':h['prefix_ids'],'bridge_ids':h['bridge_ids']},
         'prefill_chunk':512,'continuation_forward_chunk':1,'sampling':{'temperature':.6,'top_p':.95,'top_k':20},
         'eos':[151643,151645],'closing_think':151668}
        put(dest/'draw_identity.json',outer);put(dest/'sampler/identity.json',ident)
        record={'task_id':tid,'answer_seed':seed['answer_seed'],'answer_ids':[9,151645],'answer_text':'synthetic',
                'rng_final':[1,2,3],'sampler_identity_sha256':m.digest(ident)}
        resume={'state':'complete','identity_sha256':m.digest(ident),'tokens':record['answer_ids'],'forwarded_tokens':1,
          'rng_initial':[4,5,6],'rng_state':record['rng_final'],'logits_sha256':'f'*64,'record':record,'record_sha256':m.digest(record)}
        resume['resume_sha256']=m.digest(resume);put(dest/'sampler/resume.json',resume)
        ap=put(dest/'sampler/answer_record.json',record)
        put(dest/'sampler/complete.json',{'identity_sha256':m.digest(ident),'record_sha256':m.digest(record),
             'record_file_sha256':m.sha(ap),'record_file':'answer_record.json'})
        put(dest/'sampler/completion_timing.json',{'identity_sha256':m.digest(ident),'record_sha256':m.digest(record)})
        put(dest/'answer.json',{**record,**outer});put(dest/'working_set.json',{'semantic_trajectory_union_complete':True})
        cp=put(dest/'complete.json',{'identity':outer,'files':{n:m.sha(dest/n)for n in m.REQUIRED}})
        files.append({'path':str(cp.relative_to(folder)),'sha256':m.sha(cp)})
    put(folder/'task_complete.json',{'task_id':tid,'declaration_sha256':m.DECL_SHA,'implementation_sha256':m.IMPLEMENTATION_SHA,
      'completed_records':42,'expected_records':42,'backing_cache_fingerprints_unchanged':True,
      'prompt_only_template_sha256':m.sha(tp),'files':files})
    return task,d,folder

class Tests(unittest.TestCase):
 def test_complete_population_is_verified_and_not_pending(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);r=m.task_state(root,task,d,m.IMPLEMENTATION_SHA)
   self.assertTrue(r['task_complete']);self.assertEqual(r['completed_draws'],42);self.assertFalse(r['corner_cases'])
 def test_corner_is_preserved_and_flagged_not_rerolled(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);(folder/'task_complete.json').unlink();(folder/'D/seed_0/complete.json').unlink()
   before=m.sha(folder/'D/seed_0/sampler/resume.json');r=m.task_state(root,task,d,m.IMPLEMENTATION_SHA)
   self.assertFalse(r['task_complete']);self.assertEqual(r['completed_draws'],41);self.assertEqual(len(r['corner_cases']),1)
   self.assertEqual(m.sha(folder/'D/seed_0/sampler/resume.json'),before)
 def test_changed_completed_bytes_block(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);put(folder/'D/seed_0/answer.json',{})
   with self.assertRaisesRegex(ValueError,'descendant'):m.task_state(root,task,d,m.IMPLEMENTATION_SHA)
 def test_wrong_runtime_blocks(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);p=folder/'D/seed_0/sampler/identity.json';r=m.read(p);r['model']['runtime_identity_sha256']='changed';put(p,r)
   with self.assertRaisesRegex(ValueError,'identity differs'):m.task_state(root,task,d,m.IMPLEMENTATION_SHA)
 def test_independent_copy_never_rewrites_existing_or_original(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);src=root/'source';src.write_bytes(b'original');dst=root/'new/copy';expected=m.sha(src)
   self.assertEqual(m.copy_independent(src,dst,expected),'copied_independent_bytes')
   self.assertNotEqual(src.stat().st_ino,dst.stat().st_ino);self.assertEqual(src.stat().st_nlink,1)
   self.assertEqual(m.copy_independent(src,dst,expected),'identical_existing_untouched')
   dst.write_bytes(b'divergent guard')
   with self.assertRaises(ValueError):m.copy_independent(src,dst,expected)
   self.assertEqual(dst.read_bytes(),b'divergent guard');self.assertEqual(src.read_bytes(),b'original')
 def test_hardlinks_and_symlinks_block(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);src=root/'source';src.write_bytes(b'x');dst=root/'hardlink';os.link(src,dst)
   with self.assertRaisesRegex(ValueError,'hardlinked'):m.copy_independent(src,dst,m.sha(src))
   (root/'linked').symlink_to(src)
   with self.assertRaises(ValueError):m.scoped(root,'linked')
   with self.assertRaises(ValueError):m.scoped(root,'private/tests.json')
 def test_absence_requires_root_bound_observation(self):
  with tempfile.TemporaryDirectory()as t:
   p=pathlib.Path(t)/'closed.json';x={'pod_id':m.OLD_POD,'observed_absent_epoch':time.time()-5,'network_volume_preserved':True};put(p,x)
   r=m.absence_proof(p,m.sha(p),{'allocation_epoch':time.time()-10});self.assertTrue(r['provider_absence_attested_by_root_not_independently_queried_here'])
   x['pod_id']='different';put(p,x)
   with self.assertRaises(ValueError):m.absence_proof(p,m.sha(p),{'allocation_epoch':0})
 def test_tree_excludes_locks_tmp_but_retains_durable_state(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);put(root/'task/sampler/resume.json',{'synthetic':1});(root/'task/sampler/.sampler.lock').touch();(root/'task/sampler/a.tmp').touch()
   self.assertEqual(m.tree(root,'task'),['task/sampler/resume.json'])

 def test_nested_sampler_link_refused_before_private_target_read(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);p=folder/'D/seed_0/sampler/resume.json';p.unlink()
   secret=root/'private/synthetic.json';secret.parent.mkdir();secret.write_text('do not parse')
   p.symlink_to(secret)
   with self.assertRaisesRegex(ValueError,'Symlink'):m.task_state(root,task,d,m.IMPLEMENTATION_SHA)
 def test_planning_and_lock_links_cannot_mutate_old_source(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);old=root/'old';old.mkdir();dest=root/'new';(dest/m.RESULT).mkdir(parents=True)
   (dest/m.RESULT/'planning').symlink_to(old,target_is_directory=True)
   with self.assertRaisesRegex(ValueError,'Linked'):m.planning_paths(dest)
   self.assertEqual(list(old.iterdir()),[])
   (dest/m.RESULT/'planning').unlink();(dest/m.RESULT/'planning').mkdir()
   marker=old/'marker';marker.write_text('old');(dest/m.RESULT/'planning/resume_stage.lock').symlink_to(marker)
   with self.assertRaisesRegex(ValueError,'Linked'):m.planning_paths(dest)
   self.assertEqual(marker.read_text(),'old')
 def test_missing_reconstruction_fingerprint_blocks(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);task,d,folder=fixture(root);p=folder/'D/seed_0/sampler/resume.json';r=m.read(p)
   r['logits_sha256']=None;r['resume_sha256']=m.digest({k:v for k,v in r.items()if k!='resume_sha256'});put(p,r)
   with self.assertRaisesRegex(ValueError,'fingerprint'):m.task_state(root,task,d,m.IMPLEMENTATION_SHA)

 def test_source_absent_destination_transaction_is_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);rel=str(m.RESULT/'screen/tasks/task__1/D/seed_0/sampler/complete.json')
   put(root/rel,{'unreviewed_progress':True})
   with self.assertRaisesRegex(ValueError,'source-absent'):m.validate_existing_tasks(root,['task/1'],set(),{})
   self.assertTrue((root/rel).is_file())
 def test_identical_destination_task_subset_is_allowed_but_changed_is_not(self):
  with tempfile.TemporaryDirectory()as t:
   root=pathlib.Path(t);rel=str(m.RESULT/'screen/tasks/task__1/D/seed_0/sampler/resume.json')
   put(root/rel,{'synthetic_progress':True});expected=m.sha(root/rel)
   m.validate_existing_tasks(root,['task/1'],{rel},{rel:expected})
   put(root/rel,{'synthetic_progress':False})
   with self.assertRaisesRegex(ValueError,'differs'):m.validate_existing_tasks(root,['task/1'],{rel},{rel:expected})

if __name__=='__main__':unittest.main()
