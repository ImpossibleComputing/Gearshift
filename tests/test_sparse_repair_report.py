import copy
import contextlib
import io
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("report",ROOT/"scripts/sparse_repair_report.py")
report=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(report)


def fixture(complete=False):
    tids=[f"task/{i}" for i in range(12)]
    cal=[{"task_id":f"cal/{i}"} for i in range(4)]
    d={"screen_tasks":[{"task_id":t} for t in tids],"calibration_tasks":cal}
    rows=[]
    for i,tid in enumerate(tids):
        for c in report.CONDITIONS:
            for s in range(3):
                passed=(i%2==0) if c!="R_10" else (i%3!=0)
                rows.append({"task_id":tid,"condition":c,"seed_index":s,"answer_seed":s,
                    "status":"scored" if complete else "not_started","passed":passed if complete else None,
                    "score_missing":False if complete else None,"category":("pass" if passed else "test_assertion") if complete else None,
                    "answer_tokens":20 if complete else None,"answer_ended_eos":True if complete else None,
                    "answer_capped":False if complete else None,"answer_path":None,"answer_sha256":None,
                    "timing":{},"working_set":None,"memory":None,"historical_reasoning_positions":100})
    return {"declaration":d,"declaration_sha256":"frozen","draws":rows,
        "calibration_cases":[{**r,"completion_receipt_verified":True,"passed":True} for r in cal] if complete else [],
        "calibration_failures":[],"calibration_status":[],"local_probes":[],"generation_sealed":complete,
        "scoring_manifest_present":complete,"supplements":{},"unmanifested_score_receipts_not_used_for_quality":0}


def review_fixture(summary, substantive=False):
    return {"schema":"gearshift.sparse_repair.decision_review.v1", "evidence_input_sha256":"a"*64,
        "declaration_sha256":summary["declaration_sha256"], "analyst":"Synthetic test analyst",
        "reviewed_at_utc":"2026-09-21T05:00:00Z", "questions":[
            {"id":key,"verdict":("cheaper_selector" if key=="next_investment" else "supported") if substantive else "unavailable",
             "rationale":"Synthetic test interpretation, not an actual finding.",
             "limitations":["Synthetic data only."],"evidence_refs":["/coverage/scored_binary_draws"]}
            for key in report.DECISION_QUESTION_IDS]}


def cpu_stage_fixture(root):
    """Public metadata only, with no private test contents or candidate execution."""
    folder=root/"allocations/pod_test/sparse_cpu_job/job_test"
    lease_path=root/"resources/cpu_test/lease.json"
    lease={"experiment_id":"sparse_repair_01","pod_id":"pod_test","gpu_count":0,
        "control_relative":"allocations/pod_test","allocation_epoch":100.,"deadline_epoch":200.,
        "allowed_result_root":"/remote/repo/results/sparse_repair_01"}
    report.write(lease_path,lease);report.write(lease_path.parent/"pod.json",{"id":"pod_test"})
    plan={"experiment_id":"sparse_repair_01","job_id":"job_test","declaration_sha256":"declaration_sha",
        "declaration_path":"configs/declaration.json","result_root":str(report.DEFAULT_ROOT),"automatic_retries":0,
        "lease_path":"results/sparse_repair_01/resources/cpu_test/lease.json","lease_sha256":report.sha(lease_path)}
    report.write(folder/"plan.json",plan)
    for i,stage in enumerate(("preflight","prepare","run","finalize")):
        argv=["/python","scripts/sparse_repair_score.py",stage,"--repo","/remote/repo"]
        if stage in ("preflight","run"):
            argv += ["--lease",plan["lease_path"],"--lease-sha256",plan["lease_sha256"]]
        if stage in ("preflight","prepare"):
            argv += ["--declaration",plan["declaration_path"],"--declaration-sha256",plan["declaration_sha256"]]
        if stage=="prepare":argv += ["--result-root",plan["result_root"]]
        if stage in ("run","finalize"):
            argv += ["--plan",str(report.DEFAULT_ROOT/"screen/scoring/scoring_plan.json")]
        report.write(folder/(stage+"_launch.json"),{"argv":argv,"automatic_retry":False,"epoch":110.+i*10})
        if stage!="finalize":report.write(folder/(stage+"_exit.json"),{"epoch":115.+i*10,"returncode":0})
    return folder,lease_path


class ReportTests(unittest.TestCase):
    def test_final_test_timing_excludes_nested_retry_durations_and_missing_is_not_zero(self):
        score={"executed_tests":3,"total_tests":7,"receipts":[
            {"cpu_seconds":1.,"wall_seconds":2.,"attempts":[{"cpu_seconds":100.},{"cpu_seconds":1.}]},
            {"cpu_seconds":.5,"wall_seconds":1.},
            {"cpu_seconds":None,"wall_seconds":None}]}
        timing=report.score_test_accounting(score)
        self.assertEqual(timing["test_receipt_count"],3)
        self.assertEqual(timing["test_receipts_with_multiple_attempts"],1)
        self.assertEqual(timing["recorded_final_test_cpu_seconds"],{"n":2,"sum":1.5,"test_receipts_without_duration":1})
        self.assertEqual(timing["recorded_final_test_wall_seconds"]["sum"],3.)
        empty=report.score_test_accounting({"executed_tests":0})
        self.assertEqual(empty["executed_tests"],0);self.assertIsNone(empty["total_tests"])
        self.assertIsNone(empty["recorded_final_test_cpu_seconds"]["sum"])
        for bad in (float("nan"),-1,True,"1"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                report.score_test_accounting({"receipts":[{"cpu_seconds":bad}]})
        for malformed in ({"receipts":{}},{"receipts":[None]},
                          {"receipts":[{"attempts":None}]},{"executed_tests":True},{"total_tests":-1}):
            with self.subTest(malformed=malformed),self.assertRaises(ValueError):
                report.score_test_accounting(malformed)

    def test_cpu_accounting_is_separate_from_quality_and_has_explicit_coverage(self):
        data=fixture(True);baseline=report.summarize(data)
        data["draws"][0]["cpu_score_accounting"]=report.score_test_accounting({"executed_tests":1,"total_tests":4,
            "receipts":[{"cpu_seconds":.2,"wall_seconds":.3}]})
        data["draws"][1]["cpu_score_accounting"]=report.score_test_accounting({"executed_tests":0})
        summary=report.summarize(data);account=summary["cpu_scoring_accounting"]
        self.assertEqual(summary["statistics"],baseline["statistics"])
        self.assertEqual(summary["coverage"],baseline["coverage"])
        self.assertEqual(account["manifest_bound_score_records_with_accounting"],2)
        self.assertEqual(account["executed_tests"]["sum"],1)
        self.assertEqual(account["total_tests"],{"score_records_with_count":1,"score_records_without_count":1,"sum":4})
        self.assertEqual(account["recorded_final_test_cpu_seconds"]["sum"],.2)
        rendered=copy.deepcopy(summary);rendered["calibration_cases"]=[]
        text=report.render(rendered,"snapshot")
        self.assertIn("not elapsed scoring wall time",text)
        self.assertIn("Do not add either sum to wrapper elapsed time",text)
        self.assertIn("Manifest-bound score records with public test accounting: **2/504**",text)
        empty=report.cpu_accounting(fixture())
        self.assertEqual(empty["manifest_bound_score_records_with_accounting"],0)
        self.assertIsNone(empty["recorded_final_test_wall_seconds"]["sum"])

    def test_unmanifested_scores_supply_neither_quality_nor_test_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"results";declaration=Path(tmp)/"declaration.json"
            d=fixture()["declaration"];d["conditions"]=[{"name":name} for name in report.CONDITIONS]
            d["seeds"]={r["task_id"]:[{"seed_index":i,"answer_seed":i} for i in range(3)] for r in d["screen_tasks"]}
            for row in d["screen_tasks"]:row["historical_reasoning_positions"]=100
            report.write(declaration,d)
            path=root/"screen/scoring/scores/unmanifested/score.json"
            path.parent.mkdir(parents=True);path.write_text("not even valid JSON: must not be loaded")
            data=report.collect(root,declaration);summary=report.summarize(data)
            self.assertEqual(data["unmanifested_score_receipts_not_used_for_quality"],1)
            self.assertNotIn(str(path),[s["path"] for s in data["source_inputs"]])
            self.assertEqual(summary["cpu_scoring_accounting"]["manifest_bound_score_records_with_accounting"],0)
            self.assertIsNone(summary["cpu_scoring_accounting"]["recorded_final_test_cpu_seconds"]["sum"])
            self.assertIsNone(summary["statistics"])

    def test_public_cpu_stage_intervals_are_hash_bound_with_no_inferred_open_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,lease_path=cpu_stage_fixture(root);loaded=[]
            def load(path):
                loaded.append({"path":str(path),"sha256":report.sha(path)})
                return report.read(path)
            rows=report.collect_cpu_stage_intervals(root,"declaration_sha",load)
            self.assertEqual([r["stage"] for r in rows],["preflight","prepare","run","finalize"])
            self.assertEqual([r["elapsed_seconds"] for r in rows],[5.,5.,5.,None])
            self.assertEqual(rows[-1]["status"],"LAUNCH_WITHOUT_EXIT_RECEIPT")
            self.assertIsNone(rows[-1]["exit_epoch"]);self.assertIsNone(rows[-1]["exit_sha256"])
            for row in rows:
                self.assertIn({"path":row["launch_path"],"sha256":row["launch_sha256"]},loaded)
                self.assertEqual(row["lease_sha256"],report.sha(lease_path))
                self.assertFalse(row["launch_or_exit_after_lease_deadline"])
            data=fixture();data["cpu_wrapper_stage_intervals"]=rows
            summary=report.summarize(data)
            self.assertEqual(summary["cpu_scoring_accounting"]["wrapper_stage_coverage"],
                {"launch_receipts":4,"exit_receipts":3,"jobs":1})
            text=report.render(summary,"snapshot")
            self.assertIn("| pod_test / job_test | run | 5.0 | 0 |",text)
            self.assertIn("no inferred duration, completion, or current-liveness claim",text)
            # Failed/late exits remain visible measurements, not inferred success or silently discarded time.
            report.write(folder/"finalize_exit.json",{"epoch":205.,"returncode":1})
            final=report.collect_cpu_stage_intervals(root,"declaration_sha",report.read)[-1]
            self.assertTrue(final["launch_or_exit_after_lease_deadline"])
            self.assertEqual(final["elapsed_seconds"],65.);self.assertEqual(final["returncode"],1)

    def test_cpu_stage_collection_rejects_wrong_identities_and_malformed_intervals(self):
        variants=[("plan.json","job_id","different"),("plan.json","lease_sha256","wrong"),
            ("plan.json","declaration_sha256","wrong"),("plan.json","automatic_retries",1),
            ("plan.json","lease_path","/outside/lease.json"),
            ("preflight_launch.json","automatic_retry",True),
            ("preflight_launch.json","epoch",99.),("preflight_exit.json","epoch",109.),
            ("preflight_exit.json","returncode",True)]
        for filename,key,value in variants:
            with self.subTest(filename=filename,key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder,_=cpu_stage_fixture(root)
                row=report.read(folder/filename);row[key]=value;report.write(folder/filename,row)
                with self.assertRaises(ValueError):report.collect_cpu_stage_intervals(root,"declaration_sha",report.read)
        for mutation in ("pod","wrong_stage","lease_binding","exit_without_launch"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder,lease_path=cpu_stage_fixture(root)
                if mutation=="pod":report.write(lease_path.parent/"pod.json",{"id":"other_pod"})
                elif mutation=="exit_without_launch":(folder/"preflight_launch.json").unlink()
                else:
                    path=folder/"preflight_launch.json";launch=report.read(path)
                    if mutation=="wrong_stage":launch["argv"][2]="run"
                    else:launch["argv"][launch["argv"].index("--lease-sha256")+1]="wrong"
                    report.write(path,launch)
                with self.assertRaises(ValueError):report.collect_cpu_stage_intervals(root,"declaration_sha",report.read)

    def test_optional_review_keeps_numeric_summary_unchanged(self):
        summary=report.summarize(fixture());before=copy.deepcopy(summary)
        review=review_fixture(summary);review["questions"][0]["verdict"]="blocked"
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"decision_review.json";report.write(path,review)
            annotated=report.attach_decision_review(summary,path,"a"*64)
            self.assertEqual(summary,before)
            self.assertEqual({k:v for k,v in annotated.items() if k!="analyst_interpretation"},before)
            self.assertEqual(annotated["analyst_interpretation"]["review_sha256"],report.sha(path))
            text=report.render(annotated,"a"*64,"review/decision_review.json")
            self.assertIn("Analyst interpretation — not additional measurement",text)
            self.assertIn("--decision-review review/decision_review.json",text)
            self.assertIsNone(annotated["statistics"])
            self.assertIn("Unavailable: R_10−H",text)
        self.assertNotIn("Analyst interpretation",report.render(summary,"a"*64))

    def test_substantive_review_only_accepts_complete_evidence(self):
        summary=report.summarize(fixture(True));review=review_fixture(summary,True)
        report.validate_decision_review(review,summary,"a"*64)
        incomplete=report.summarize(fixture())
        with self.assertRaisesRegex(ValueError,"all 504 binary outcomes"):
            report.validate_decision_review(review,incomplete,"a"*64)
        changes=[("generation_sealed",False),("scoring_manifest_present",False),("statistics",None),
                 ("evidence_integrity_issues",[{"path":"bad","issues":["mismatch"]}])]
        for key,value in changes:
            changed=copy.deepcopy(summary);changed[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"all 504 binary outcomes"):
                report.validate_decision_review(review,changed,"a"*64)
        for key,value in (("scored_binary_draws",503),("generated_complete_draws",503),
                          ("all_four_calibration_tasks_have_verified_pass",False),
                          ("calibration_passed_task_ids",["cal/0","cal/1","cal/2"])):
            changed=copy.deepcopy(summary);changed["coverage"][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"all 504 binary outcomes"):
                report.validate_decision_review(review,changed,"a"*64)
        changed=copy.deepcopy(summary);changed["condition_diagnostics"]["D"]["scored_binary"]=35
        with self.assertRaisesRegex(ValueError,"all 504 binary outcomes"):
            report.validate_decision_review(review,changed,"a"*64)

    def test_review_requires_exact_snapshot_and_declaration_bindings(self):
        summary=report.summarize(fixture());review=review_fixture(summary)
        for field,value in (("evidence_input_sha256","b"*64),("declaration_sha256","different")):
            changed=copy.deepcopy(review);changed[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"hash binding mismatch"):
                report.validate_decision_review(changed,summary,"a"*64)

    def test_review_requires_exact_five_ids_and_valid_verdicts(self):
        summary=report.summarize(fixture());review=review_fixture(summary)
        variants=[review["questions"][:-1],review["questions"]+[review["questions"][0]],
                  list(reversed(review["questions"])),[review["questions"][0]]*5]
        for questions in variants:
            with self.assertRaisesRegex(ValueError,"five fixed question IDs"):
                report.validate_decision_review({**review,"questions":questions},summary,"a"*64)
        changed=copy.deepcopy(review);changed["questions"][0]["verdict"]="cheaper_selector"
        with self.assertRaisesRegex(ValueError,"Invalid decision verdict"):
            report.validate_decision_review(changed,summary,"a"*64)

    def test_review_json_pointers_handle_escaping_and_reject_invalid_refs(self):
        self.assertEqual(report.summary_pointer({"a/b":{"~key":[7]}},"/a~1b/~0key/0"),7)
        summary=report.summarize(fixture());review=review_fixture(summary)
        for pointer in ("", "coverage", "/absent", "/statistics/missing", "/coverage/~2bad",
                        "/limitations/-", "/limitations/00", "/limitations/999"):
            changed=copy.deepcopy(review);changed["questions"][0]["evidence_refs"]=[pointer]
            with self.subTest(pointer=pointer),self.assertRaises(ValueError):
                report.validate_decision_review(changed,summary,"a"*64)
        changed=copy.deepcopy(review);changed["questions"][0]["evidence_refs"]=["/statistics"]
        report.validate_decision_review(changed,summary,"a"*64)  # Null is valid evidence of unavailability.

    def test_review_requires_attribution_rationale_limits_and_refs(self):
        summary=report.summarize(fixture());review=review_fixture(summary)
        for key,value in (("analyst",""),("reviewed_at_utc","2026-09-21T05:00:00"),("schema","wrong")):
            changed=copy.deepcopy(review);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                report.validate_decision_review(changed,summary,"a"*64)
        for key,value in (("rationale",""),("limitations",[]),("evidence_refs",[]),
                          ("evidence_refs",["/status","/status"]),("evidence_refs",[{}])):
            changed=copy.deepcopy(review);changed["questions"][0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                report.validate_decision_review(changed,summary,"a"*64)

    def test_portable_review_cli_is_explicit_and_does_not_change_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);source=base/"input_records.json";manifest=base/"input_manifest.json"
            output=base/"output";md=base/"report.md";review_path=base/"decision_review.json"
            report.write(source,fixture());input_sha=report.sha(source)
            report.write(manifest,{"file":source.name,"sha256":input_sha})
            numeric_bytes=source.read_bytes();manifest_bytes=manifest.read_bytes()
            summary=report.summarize(fixture());review=review_fixture(summary);review["evidence_input_sha256"]=input_sha
            report.write(review_path,review)
            argv=["report.py","--input",str(source),"--output",str(output),"--report",str(md)]
            with mock.patch("sys.argv",argv),contextlib.redirect_stdout(io.StringIO()):report.main()
            plain=report.read(output/"summary.json")
            self.assertNotIn("analyst_interpretation",plain)
            with mock.patch("sys.argv",argv+["--decision-review",str(review_path)]),contextlib.redirect_stdout(io.StringIO()):report.main()
            annotated=report.read(output/"summary.json")
            self.assertEqual({k:v for k,v in annotated.items() if k!="analyst_interpretation"},plain)
            self.assertEqual(source.read_bytes(),numeric_bytes);self.assertEqual(manifest.read_bytes(),manifest_bytes)
            self.assertEqual(annotated["analyst_interpretation"]["review"]["evidence_input_sha256"],input_sha)
            review["evidence_input_sha256"]="b"*64;report.write(review_path,review)
            saved_output=(output/"summary.json").read_bytes()
            with mock.patch("sys.argv",argv+["--decision-review",str(review_path)]),self.assertRaisesRegex(ValueError,"hash binding"):
                report.main()
            self.assertEqual((output/"summary.json").read_bytes(),saved_output)

    def test_zero_of_504_is_not_quality_failure(self):
        s=report.summarize(fixture())
        self.assertEqual(s["coverage"]["generated_complete_draws"],0)
        self.assertEqual(s["coverage"]["not_generated_complete_draws"],504)
        self.assertIsNone(s["statistics"])
        self.assertIsNone(s["condition_diagnostics"]["H"]["passed"])
        self.assertIsNone(s["condition_diagnostics"]["H"]["pass_rate_full_population"])
        text=report.render(s,"test")
        self.assertIn("Unrun draws are **not** quality failures",text)
        self.assertIn("Unavailable: R_10−H",text)

    def test_explanatory_limits_distinguish_sparse_interventions(self):
        text=report.render(report.summarize(fixture()),"test")
        for expected in ("Sparse training-loss positions, sparse attention reads, and sparse KV construction",
                         "earlier-token dependencies", "corresponding raw token alone",
                         "share a selector rule, not necessarily selected indices",
                         "not guaranteed identical to D", "No jacq kernel or serving speedup"):
            self.assertIn(expected,text)

    def test_calibration_launch_does_not_infer_screen_launch_or_quality(self):
        data=fixture();data["supplements"]={"preflight/gpu_launch.json":{"status":"GPU_WORK_LAUNCHED"}}
        value=report.summarize(data);text=report.render(value,"test")
        self.assertIn("A calibration launch is not evidence",text)
        self.assertIn("`screen_launch.json`: unavailable",text)
        self.assertIn("`screen_all_workers_generating.json`: unavailable",text)
        self.assertIsNone(value["statistics"])

    def test_screen_observations_are_historical_progress_not_quality(self):
        data=fixture()
        generating={"role":"sparse_repair_screen","stage":"answer_generation","generated_tokens":65}
        data["supplements"]={"preflight/screen_launch.json":{"epoch":10,"status":"SCREEN_GENERATION_LAUNCHED",
            "workers":[{"name":"one","worker_status":generating},
                       {"name":"two","worker_status":{"stage":"starting"}}]},
            "preflight/screen_all_workers_generating.json":{"epoch":20,"all_five_actual_generation_observed":True,
                "observations":[{"name":str(i),"worker_status":generating} for i in range(5)]}}
        value=report.summarize(data);text=report.render(value,"test")
        self.assertIn("1/2 named workers",text)
        self.assertIn("5/5 named workers",text)
        self.assertIn("not a current liveness check or a completion seal",text)
        self.assertEqual(value["coverage"]["generated_complete_draws"],0)
        self.assertIsNone(value["statistics"])
        self.assertIn("Unavailable: R_10−H",text)

    def test_partial_outcome_never_bootstraps_favorable_subset(self):
        d=fixture();d["draws"][0].update(status="scored",passed=True,score_missing=False,answer_tokens=5)
        s=report.summarize(d)
        self.assertIsNone(s["statistics"])
        self.assertEqual(s["condition_diagnostics"]["D"]["passed"],1)
        self.assertIsNone(s["condition_diagnostics"]["D"]["pass_rate_full_population"])

    def test_all12clusters_and_same_joint_paired_bootstrap(self):
        d=fixture(True);s=report.summarize(d)
        stats=s["statistics"]
        self.assertEqual(stats["conditions"]["H"]["passed"],18)
        self.assertEqual(stats["conditions"]["R_10"]["passed"],24)
        self.assertAlmostEqual(stats["contrasts"]["R_10-H"]["difference"],1/6)
        self.assertEqual(len(stats["per_task"]),12*8)
        again=report.cluster_statistics(d["draws"],[x["task_id"] for x in d["declaration"]["screen_tasks"]])
        self.assertEqual(stats,again)

    def test_missing_calibration_blocks_complete_claim(self):
        d=fixture(True);d["calibration_cases"].pop()
        s=report.summarize(d)
        self.assertIsNone(s["statistics"])
        self.assertNotEqual(s["status"],"COMPLETE_EXPLORATORY_SCREEN")

    def test_missing_or_duplicate_draw_rejected(self):
        d=fixture();d["draws"].pop()
        with self.assertRaisesRegex(ValueError,"population"):
            report.summarize(d)
        d=fixture();d["draws"][-1]=copy.deepcopy(d["draws"][0])
        with self.assertRaisesRegex(ValueError,"population"):
            report.summarize(d)

    def test_cluster_statistics_rejects_missing_outcomes(self):
        d=fixture(True);d["draws"][0]["passed"]=None
        with self.assertRaisesRegex(ValueError,"binary outcomes"):
            report.cluster_statistics(d["draws"],[x["task_id"] for x in d["declaration"]["screen_tasks"]])

    def test_working_set_counts_layer_head_pairs_not_global_token_union(self):
        row=fixture()["draws"][0]
        row["working_set"]={"layers":{"0":{"cumulative_pairs_by_kv_head":[50,60],
            "peak_instantaneous_selected_pairs":20,"cumulative_per_head_page_bytes_estimated":256}},
            "cumulative_unique_pairs_all_layers":110,"cumulative_logical_selected_bytes_all_layers":110*512,
            "frozen_native_history_bytes_exact":204800,"frozen_mapped_history_bytes_exact":204800}
        w=report.working_set(row)
        self.assertEqual(w["reasoning_pair_capacity"],200)
        self.assertEqual(w["cumulative_reasoning_fraction"],.55)
        self.assertFalse(w["physical_memory_traffic_measured"])

    def test_dense_zero_selector_counter_is_not_zero_dense_reads(self):
        row=fixture()["draws"][0]
        row["working_set"]={"mode":"disabled","fraction":1.0,"layers":{},
            "cumulative_unique_pairs_all_layers":0,"cumulative_logical_selected_bytes_all_layers":0,
            "frozen_native_history_bytes_exact":1234}
        value=report.working_set(row)
        self.assertEqual(value["controller_selected_pair_counter_raw"],0)
        self.assertFalse(value["selected_pair_accounting_applicable"])
        self.assertIsNone(value["cumulative_unique_pairs"])
        self.assertIsNone(value["cumulative_logical_selected_bytes"])
        self.assertIsNone(value["cumulative_reasoning_fraction"])
        self.assertEqual(value["native_history_bytes"],1234)

    def test_per_head_spread_page_layouts_and_buffers_remain_distinct(self):
        row=fixture()["draws"][0]
        row["working_set"]={"fraction":.1,"layers":{"0":{"cumulative_pairs_by_kv_head":[50,60],
            "peak_instantaneous_selected_pairs":20,"cumulative_per_head_page_bytes_estimated":256,
            "cumulative_all_heads_shared_page_bytes_estimated":512,"cumulative_union_metadata_bytes":200}},
            "cumulative_unique_pairs_all_layers":110,"cumulative_logical_selected_bytes_all_layers":110*512,
            "per_layer_peak_buffer_sizes":{"0":{"kv_group_allowed_mask_bytes_exact":1000}}}
        row["shared_task_backing_bytes"]={"source_peak_then_freed":9999,"native":1234}
        value=report.working_set(row)
        self.assertEqual(value["cumulative_pair_fraction_by_layer_and_kv_head"],{"0":[.5,.6]})
        self.assertEqual(value["peak_instantaneous_reasoning_pair_fraction"],.1)
        self.assertEqual(value["cumulative_per_head_page_bytes_estimated"],256)
        self.assertEqual(value["cumulative_all_heads_shared_page_bytes_estimated"],512)
        self.assertEqual(value["cumulative_union_metadata_bytes"],200)
        self.assertEqual(value["per_layer_peak_buffer_sizes_logical_tensor_bytes"]["0"]["kv_group_allowed_mask_bytes_exact"],1000)
        self.assertEqual(value["shared_task_backing_bytes"]["source_peak_then_freed"],9999)
        self.assertFalse(value["physical_memory_traffic_measured"])

    def test_completion_overhead_is_preserved_not_added_to_active_time(self):
        data=fixture();data["draws"][0].update(status="generated_unscored",answer_tokens=10,
            timing={"answer_seconds":10,"this_attempt_whole_draw_wall_seconds":12,
                    "sampler_completion_timing":{"overhead":{"scope_complete":True,
                        "result_materialization_seconds":.3,"progress_publish_seconds":.8}}})
        value=report.summarize(data);arm=value["condition_diagnostics"]["D"]
        self.assertEqual(arm["answer_active_seconds"]["mean"],10)
        self.assertEqual(arm["actual_attempt_whole_draw_seconds"]["mean"],12)
        self.assertEqual(arm["sampler_completion_overhead_seconds"]["result_materialization_seconds"]["mean"],.3)
        self.assertEqual(arm["sampler_completion_overhead_scope_complete"],1)
        text=report.render(value,"test")
        for expected in ("not a global token union", "current/own-answer states", "zero dense reads",
                         "hypothetical 16-position layouts", "query-head-expanded mask", "source_peak_then_freed",
                         "before the final working-set/answer/complete JSON writes", "must not be added again"):
            self.assertIn(expected,text)

    def test_final_complete_coverage_keeps_prior_interruption_without_changing_outcomes(self):
        data=fixture(True)
        for case in data["calibration_cases"]:case["path"]="synthetic/calibration.json"
        baseline=report.summarize(data)
        data["supplements"]={"recovery/decision.json":{"global_complete_answers":476,"expected_answers":504,
            "pending_task_ids":["task/0"],"complete_draws_in_pending_task":14,
            "interrupted_draw":{"condition":"R_recent","seed_index":2,"committed_tokens":109,"forwarded_tokens":108},
            "reason":"Synthetic deadline interruption, not an outcome retry."},
            "recovery/resume_stage.json":{"status":"STAGED_NO_LAUNCH","model_execution_performed":False},
            "recovery/resume_launch.json":{"status":"PROCESS_LAUNCHED"}}
        data["original_interrupted_sampler_metadata"]=[{"identity":{"task_id":"task/0","condition":"R_recent","seed_index":2},
            "state":"interrupted","committed_tokens":109,"forwarded_tokens":108,"timing":{"active_seconds":20.}}]
        data["task_cache_setup_attempts"]=[{"task_id":"task/0","timing":{"mapper_seconds":1.}},
                                            {"task_id":"task/0","timing":{"mapper_seconds":1.1}}]
        data["draws"][0]["timing"]={"sampler_timing":{"resume_count":1,"reconstruction_seconds":2.},
            "trace_rebuild_of_completed_sampler_seconds":0.,"this_attempt_whole_draw_wall_seconds":30.}
        value=report.summarize(data);text=report.render(value,"synthetic")
        self.assertEqual(value["coverage"],baseline["coverage"])
        self.assertEqual(value["statistics"],baseline["statistics"])
        self.assertEqual(value["status"],"COMPLETE_EXPLORATORY_SCREEN")
        self.assertIn("**504/504**",text);self.assertIn("**476/504**",text)
        self.assertIn("109 committed tokens and 108 forwarded tokens",text)
        self.assertIn("STAGED_NO_LAUNCH",text)
        self.assertIn("`recovery/recovery_validation.json`: unavailable",text)
        self.assertIn('"task/0": 2',text)
        self.assertIn("| task/0 / D / 0 | 1 | 2.0 | 0.0 | 30.0 |",text)
        self.assertIn("discarded/uncommitted crash-tail selections and timing may remain incomplete",text)
        self.assertIn("last task/draw-attempt wall does not include all earlier interrupted attempts",text)
        self.assertIn("not inference from a successful clone, stage, launch or final count",text)

    def test_recovery_stage_only_does_not_claim_replay_or_completion(self):
        data=fixture();data["supplements"]={"recovery/resume_stage.json":{
            "status":"STAGED_NO_LAUNCH","launch_authorized_by_this_receipt":False,"model_execution_performed":False}}
        value=report.summarize(data);text=report.render(value,"synthetic")
        self.assertIsNone(value["statistics"])
        self.assertIn("Staging does not imply launch",text)
        self.assertIn("`recovery/resume_launch.json`: unavailable",text)
        self.assertIn("No completed resumed-draw timing is available yet",text)
        self.assertEqual(value["coverage"]["generated_complete_draws"],0)

    def test_no_zero_filling_unmeasured_metrics(self):
        self.assertEqual(report.metric([None,float("nan")]),{"n":0,"mean":None,"min":None,"max":None})

    def test_integrity_blocker_cannot_produce_complete_quality_claim(self):
        data=fixture(True);data["evidence_integrity_issues"]=[{"path":"bad.json","issues":["changed implementation"]}]
        value=report.summarize(data)
        self.assertEqual(value["status"],"EVIDENCE_INTEGRITY_BLOCKER")
        self.assertIsNone(value["statistics"])
        for case in value["calibration_cases"]:case["path"]="test/"+case["task_id"]
        self.assertIn("Evidence-integrity blocker",report.render(value,"test"))

    def test_calibration_report_gate_checks_metrics_runtime_and_implementation(self):
        row={"answer_position":0,"max_abs":0.0,"top1_equal":True,"bit_exact":True,"passed":True}
        case={"task_id":"cal/0","declaration_sha256":"d","probe_positions":[0],"implementation_hashes":{"impl":"h"},
            "runtime":{"torch":"2.8.0+cu128","cuda":"12.8","transformers":"4.57.6","gpu":"NVIDIA H200", "BF16":True,"TF32":False,"deterministic_algorithms":True},
            "controls":{name:{"positions":[dict(row)],"altered_shape":False,"passed":True} for name in report.REQUIRED_CONTROLS},
            "all_same_path_controls_passed":True,"backing_caches_unchanged":True,"passed":True}
        contract={"declaration_sha256":"d","calibration_task_ids":["cal/0"],"implementation_hashes":{"impl":"h"}}
        self.assertEqual(report.calibration_integrity(case,contract,{"impl":"h"}),[])
        changed=copy.deepcopy(case);changed["controls"][report.REQUIRED_CONTROLS[0]]["positions"][0]["max_abs"]=.5
        self.assertTrue(any("pass_flag" in x for x in report.calibration_integrity(changed,contract,{"impl":"h"})))
        self.assertIn("calibration_does_not_bind_current_numerical_implementation",report.calibration_integrity(case,contract,{"impl":"new"}))
        changed=copy.deepcopy(case);changed["runtime"]["TF32"]=True
        self.assertIn("not_pinned_actual_H200_runtime",report.calibration_integrity(changed,contract,{"impl":"h"}))

    def test_probe_reference_and_coverage_mismatches_are_explicit(self):
        task={"task_id":"cal/0","calibration_seed":7}
        contract={"declaration_sha256":"d","calibration_task_ids":["cal/0"],"implementation_hashes":{"impl":"h"}}
        trace={"task_id":"cal/0","declaration_sha256":"d","kind":"calibration_only_not_screen","complete":True,"seed":7,"answer_ids":[1]}
        rows=[{"layer":layer,"answer_position":0,"mode":mode,"fraction":p,"identical_prefix_and_current_query":True,
               "attention_output_relative_l2":.1,"attention_output_max_abs":.2,
               "selected_mass_sum_per_kv_group":[.1]*8,"all_reasoning_mass_sum_per_kv_group":[.5]*8}
              for layer in range(36) for mode in ("N","M","R") for p in (.05,.1,.25)]
        self.assertEqual(report.probe_integrity(rows,"same_query_attention",task,"d",contract,trace,{"impl":"h"}),[])
        self.assertIn("same_query_probe_coverage_or_identity_mismatch",report.probe_integrity(rows[:-1],"same_query_attention",task,"d",contract,trace,{"impl":"h"}))
        wrong=dict(trace,task_id="another/task")
        self.assertIn("probe_reference_not_bound_to_own_completed_calibration_D_trace",report.probe_integrity(rows,"same_query_attention",task,"d",contract,wrong,{"impl":"h"}))

    def test_independent_prompt_failed_shape_is_reported_not_renamed_pass(self):
        data=fixture();data["calibration_cases"]=[{"task_id":"cal/0","path":"run/calibration.json","completion_receipt_verified":True,"passed":True,
            "independent_prompt_shape_rounding":{"passed":False,"positions":[{"max_abs":.75,"kl":.000001,"top1_equal":True}]}}]
        text=report.render(report.summarize(data),"test")
        self.assertIn("| cal/0 | False | 0.75",text)
        self.assertIn("not claimed identical to full native replay D",text)
        self.assertIn("never accepted by widening the tolerance",text)

    def test_collection_uses_raw_runs_not_duplicate_canonical_gate_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"results";declaration=Path(tmp)/"declaration.json"
            d=fixture()["declaration"]
            d["conditions"]=[{"name":name} for name in report.CONDITIONS]
            d["seeds"]={r["task_id"]:[{"seed_index":i,"answer_seed":i} for i in range(3)] for r in d["screen_tasks"]}
            d["calibration_seeds"]={r["task_id"]:7 for r in d["calibration_tasks"]}
            for row in d["screen_tasks"]:row["historical_reasoning_positions"]=100
            for row in d["calibration_tasks"]:row["history_sha256"]="recorded"
            report.write(declaration,d)
            supplement=root/"preflight/screen_launch.json"
            report.write(supplement,{"status":"SCREEN_GENERATION_LAUNCHED","workers":[]})
            all_generating=root/"preflight/screen_all_workers_generating.json"
            report.write(all_generating,{"observations":[]})
            initial=report.collect(root,declaration)
            self.assertEqual(initial["supplements"]["preflight/screen_launch.json"]["status"],"SCREEN_GENERATION_LAUNCHED")
            self.assertEqual(initial["supplements"]["preflight/screen_all_workers_generating.json"],{"observations":[]})
            for path in (supplement,all_generating):
                self.assertIn({"path":str(path),"bytes":path.stat().st_size,"sha256":report.sha(path)},initial["source_inputs"])
            case={"task_id":"cal/0","history_sha256":"recorded","declaration_sha256":report.sha(declaration),"passed":True}
            canonical=root/"calibration/cal__0/calibration.json";report.write(canonical,case)
            self.assertEqual(len(report.collect(root,declaration)["calibration_cases"]),0)
            raw=root/"calibration_runs/allocation/cal__0/calibration.json";report.write(raw,case)
            report.write(raw.parent/"complete.json",{"calibration_sha256":report.sha(raw),"declaration_sha256":report.sha(declaration),"passed":True})
            data=report.collect(root,declaration)
            self.assertEqual(len(data["calibration_cases"]),1)
            self.assertTrue(data["evidence_integrity_issues"])
            self.assertFalse(data["calibration_cases"][0]["evidence_integrity_verified"])

    def test_collection_keeps_hash_bound_task_wall_and_final_sampler_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"results";declaration=Path(tmp)/"declaration.json"
            d=fixture()["declaration"];d["conditions"]=[{"name":name} for name in report.CONDITIONS]
            d["seeds"]={r["task_id"]:[{"seed_index":i,"answer_seed":i} for i in range(3)] for r in d["screen_tasks"]}
            d["calibration_seeds"]={r["task_id"]:7 for r in d["calibration_tasks"]}
            for row in d["screen_tasks"]:row["historical_reasoning_positions"]=100
            for row in d["calibration_tasks"]:row["history_sha256"]="recorded"
            report.write(declaration,d);declaration_sha=report.sha(declaration)
            folder=root/"screen/tasks/task__0";files=[]
            for condition in report.CONDITIONS:
                for seed in d["seeds"]["task/0"]:
                    dest=folder/condition/f"seed_{seed['seed_index']}"
                    identity={"task_id":"task/0","condition":condition,**seed,"declaration_sha256":declaration_sha}
                    report.write(dest/"answer.json",{**identity,"answer_ids":[1],"answer_ended_eos":True,"answer_capped":False})
                    report.write(dest/"working_set.json",{"summary":{},"semantic_trajectory_union_complete":True})
                    report.write(dest/"sampler/completion_timing.json",{"state":"complete","overhead":{"scope_complete":True,"result_materialization_seconds":.2}})
                    report.write(dest/"complete.json",{"identity":identity,"files":{name:report.sha(dest/name)
                        for name in ("answer.json","working_set.json","sampler/completion_timing.json")}})
                    files.append({"path":str((dest/"complete.json").relative_to(folder)),"sha256":report.sha(dest/"complete.json")})
            receipt={"task_id":"task/0","declaration_sha256":declaration_sha,"completed_records":42,"expected_records":42,
                "backing_cache_fingerprints_unchanged":True,"files":files,"cache_setup_timing":{"mapper_seconds":1.},
                "this_attempt_whole_task_wall_seconds":100.}
            report.write(folder/"task_complete.json",receipt)
            for attempt in ("original","continuation"):
                report.write(folder/"cache_setup_attempts"/(attempt+".json"),{
                    "task_id":"task/0","declaration_sha256":declaration_sha,"timing":{"mapper_seconds":1.},"backing_bytes":{"native":5}})
            data=report.collect(root,declaration);self.assertEqual(len(data["task_execution_records"]),1)
            self.assertEqual(len(data["task_cache_setup_attempts"]),2)
            self.assertEqual(data["task_execution_records"][0]["this_attempt_whole_task_wall_seconds"],100.)
            self.assertEqual(data["draws"][0]["timing"]["sampler_completion_timing"]["overhead"]["result_materialization_seconds"],.2)
            text=report.render(report.summarize(data),report.sha(declaration))
            self.assertIn("| task/0 | 42 | 100.0 | 1.0 |",text)
            self.assertIn("counted once rather than 42 times",text)
            receipt["files"][0]["sha256"]="bad";report.write(folder/"task_complete.json",receipt)
            with self.assertRaisesRegex(ValueError,"Task timing receipt draw hash mismatch"):
                report.collect(root,declaration)

    def test_collect_recovery_receipts_and_metadata_are_hash_bound_without_token_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"results";declaration=Path(tmp)/"declaration.json"
            d=fixture()["declaration"];d["conditions"]=[{"name":name} for name in report.CONDITIONS]
            d["seeds"]={r["task_id"]:[{"seed_index":i,"answer_seed":i} for i in range(3)] for r in d["screen_tasks"]}
            d["calibration_seeds"]={r["task_id"]:7 for r in d["calibration_tasks"]}
            for row in d["screen_tasks"]:row["historical_reasoning_positions"]=100
            for row in d["calibration_tasks"]:row["history_sha256"]="recorded"
            report.write(declaration,d)
            paths=[]
            for name in ("decision.json","resume_stage.json","resume_launch.json","recovery_validation.json"):
                path=root/"recovery"/name;report.write(path,{"status":"synthetic"});paths.append(path)
            draw=root/"recovery/original_interrupted_state/R_recent/seed_2"
            report.write(draw/"draw_identity.json",{"task_id":"task/0","condition":"R_recent","seed_index":2,
                "declaration_sha256":report.sha(declaration)})
            resume=draw/"sampler/resume.json";paths.append(resume)
            report.write(resume,{"state":"interrupted","tokens":[5]*109,"forwarded_tokens":108,
                "rng_state":[1,2,3],"timing":{"active_seconds":20.},"logits_sha256":"saved","resume_sha256":"bound"})
            report.write(draw/"failure.json",{"exception":"DeadlineReached","error":"stop margin","this_attempt_wall_seconds":21.})
            report.write(draw/"progress_working_set.json",{"summary":{"cumulative_unique_pairs_all_layers":10},
                "trace_records":{"not_compact":"must not be copied"}})
            data=report.collect(root,declaration)
            for path in paths:
                self.assertIn({"path":str(path),"bytes":path.stat().st_size,"sha256":report.sha(path)},data["source_inputs"])
            self.assertEqual(len(data["original_interrupted_sampler_metadata"]),1)
            metadata=data["original_interrupted_sampler_metadata"][0]
            self.assertEqual(metadata["committed_tokens"],109);self.assertEqual(metadata["forwarded_tokens"],108)
            self.assertEqual(metadata["failure"]["exception"],"DeadlineReached")
            self.assertNotIn("tokens",metadata);self.assertNotIn("rng_state",metadata)
            self.assertNotIn("trace_records",metadata)
            self.assertEqual(metadata["interrupted_progress_summary_not_complete_answer_union"]["cumulative_unique_pairs_all_layers"],10)
            summary=report.summarize(data);self.assertEqual(summary["coverage"]["generated_complete_draws"],0)

    def test_csv_lf_and_nested_serialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"data.csv"
            report.csv_write(path,[{"x":1,"nested":{"passed":False}}])
            self.assertNotIn(b"\r",path.read_bytes())


if __name__=="__main__":
    unittest.main()
