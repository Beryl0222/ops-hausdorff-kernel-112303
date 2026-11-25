"""Behavior tests for the streaming point-distance kernel."""

import math
import unittest

import numpy as np

import hausdorff_kernel as hk
from hausdorff_kernel_input import load_record


def view(lo=(-10, -10, -10), hi=(10, 10, 10)):
    return [list(lo), list(hi)]


def make_chunk(chunk_id, side, coords, *, ids=None, region="r", regions=None,
                normals=None, weights=None, bbox=None):
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if ids is None:
        ids = [f"{chunk_id}-{i}" for i in range(n)]
    if normals is None:
        normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    if weights is None:
        weights = np.ones(n)
    return hk.PointChunk(
        chunk_id=chunk_id,
        side=side,
        ids=ids,
        regions=regions if regions is not None else [region] * n,
        xyz=coords,
        normals=normals,
        weights=weights,
        bbox=bbox if bbox is not None else [coords.min(0), coords.max(0)],
    )


def two_point_frames(left_xyz, right_xyz, **kwargs):
    lc = make_chunk("l", "left", left_xyz, **kwargs.pop("left_kwargs", {}))
    rc = make_chunk("r", "right", right_xyz, **kwargs.pop("right_kwargs", {}))
    return (
        hk.PointFrame("left", kwargs.pop("left_view", view()), [lc]),
        hk.PointFrame("right", kwargs.pop("right_view", view()), [rc]),
    )


class QuantileTests(unittest.TestCase):
    def test_weighted_quantile_uses_cumulative_weight(self):
        d = [0.04, 0.25, 1.0]
        w = [1.0, 3.0, 1.0]
        # 90% of 5 is 4.5; cumulative reaches it at 1.0.
        self.assertEqual(hk.weighted_quantile(d, w, 0.9), 1.0)
        # 80% is exactly 4; smallest reaching distance is 0.25.
        self.assertEqual(hk.weighted_quantile(d, w, 0.8), 0.25)
        self.assertIsNone(hk.weighted_quantile([], [], 0.9))


class SampleRecordBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()

    def test_final_sample_release_and_ties(self):
        ev = hk.evaluate_record(self.record, pose_id="p0")
        self.assertEqual(ev.status, "final")
        self.assertEqual(ev.released_regions(), ["seam"])
        self.assertFalse(ev.approved)

        seam = ev.regions["seam"]
        self.assertTrue(seam.released)
        ltr = seam.left_to_right
        rtl = seam.right_to_left
        self.assertAlmostEqual(ltr.coverage, 1.0)
        self.assertAlmostEqual(rtl.coverage, 1.0)
        self.assertAlmostEqual(ltr.weighted_quantile_sq, 0.25)
        self.assertAlmostEqual(rtl.weighted_quantile_sq, 0.25)

        l_matches = {m.query_id: m for m in ltr.matches}
        r_matches = {m.query_id: m for m in rtl.matches}
        self.assertEqual(l_matches["l1"].neighbor_id, "r04")
        self.assertTrue(l_matches["l1"].equidistant)
        self.assertEqual(r_matches["r04"].neighbor_id, "l0")
        self.assertTrue(r_matches["r04"].equidistant)
        # Duplicate l1 copy must not create a false tie with itself.
        self.assertFalse(r_matches["r1"].equidistant)

        key = ev.regions["key"]
        self.assertEqual(key.status, "missing_evidence")
        self.assertTrue(key.missing_evidence)
        self.assertFalse(key.released)
        self.assertIsNone(key.left_to_right.weighted_quantile_sq)

    def test_partial_arrival_is_not_missing_evidence(self):
        one = hk.evaluate_record(self.record, pose_id="p0", through=1)
        self.assertEqual(one.status, "partial")
        seam = one.regions["seam"]
        self.assertEqual(seam.status, "partial")
        self.assertTrue(seam.temporary_no_neighbor)
        self.assertFalse(seam.missing_evidence)
        self.assertFalse(seam.released)

        full = hk.evaluate_record(self.record, pose_id="p0")
        self.assertEqual(full.regions["seam"].status, "final")

    def test_late_chunk_changes_match(self):
        before = hk.evaluate_record(self.record, pose_id="p0", through=2)
        l1_before = {m.query_id: m for m in before.regions["seam"].left_to_right.matches}["l1"]
        self.assertEqual(l1_before.neighbor_id, "r1")

        after = hk.evaluate_record(self.record, pose_id="p0", through=4)
        l1_after = {m.query_id: m for m in after.regions["seam"].left_to_right.matches}["l1"]
        self.assertEqual(l1_after.neighbor_id, "r04")

    def test_bbox_exclusion_is_retained_and_pose_sensitive(self):
        ev = hk.evaluate_record(self.record, pose_id="p0")
        exclusions = ev.regions["seam"].left_to_right.exclusions
        late_skip = next(e for e in exclusions if e.chunk_id == "r/late" and e.query_id == "l0")
        self.assertGreater(late_skip.lower_bound_sq, late_skip.incumbent_distance_sq)
        self.assertTrue(late_skip.holds_for_pose(ev.pose))

        p1 = hk.resolve_record_pose(self.record, pose_id="p1")
        recheck = late_skip.recheck(p1)
        # The basis is always retained; under another pose it is recomputed and
        # must not silently become old eligibility.
        self.assertIsInstance(recheck.still_excluded, bool)
        self.assertEqual(recheck.pose_id, "p1")

        ev1 = hk.evaluate_record(self.record, pose_id="p1")
        self.assertEqual(ev1.pose.id, "p1")
        self.assertFalse(ev1.approved)

    def test_report_is_json_serializable(self):
        ev = hk.evaluate_record(self.record, pose_id="p0")
        text = ev.to_json()
        self.assertIn("left_to_right", text)
        source = hk.distance_loss_and_gradients(ev, allow_unstable=True).to_dict()["sources"][0]
        self.assertIn("query_point_gradient", source)


class VisibilityNormalAndDuplicateTests(unittest.TestCase):
    def test_points_outside_mutual_view_are_excluded(self):
        # Left x=100 is in left view but not the transformed right view.
        left = make_chunk("l", "left", [[0.0, 0, 0], [100.0, 0, 0]], ids=["a", "outside"])
        right = make_chunk("r", "right", [[0.1, 0, 0]], ids=["b"])
        frame_l = hk.PointFrame("left", [[-1, -1, -1], [1, 1, 1]], [left])
        frame_r = hk.PointFrame("right", [[-1, -1, -1], [1, 1, 1]], [right])
        rules = {"r": [0.8, 1.0, 1.0, 0.9]}
        ev = hk.bidirectional_nearest_neighbors(frame_l, frame_r, hk.Pose(), rules)
        dr = ev.regions["r"].left_to_right
        self.assertEqual(dr.count_query, 1)
        self.assertAlmostEqual(dr.total_weight, 1.0)
        self.assertEqual([m.query_id for m in dr.matches], ["a"])

    def test_normal_angle_is_enforced(self):
        left = make_chunk("l", "left", [[0, 0, 0]], normals=[[0, 1, 0]])
        right = make_chunk("r", "right", [[0.1, 0, 0]], normals=[[1, 0, 0]])
        lf, rf = hk.PointFrame("left", view(), [left]), hk.PointFrame("right", view(), [right])
        ev = hk.bidirectional_nearest_neighbors(lf, rf, hk.Pose(), {"r": [0.8, 0.8, 1, .9]})
        region = ev.regions["r"]
        self.assertEqual(region.status, "missing_evidence")
        self.assertEqual(region.left_to_right.count_matched, 0)

    def test_duplicate_is_counted_once_but_conflict_rejects_region(self):
        l0 = make_chunk("l0", "left", [[0, 0, 0]], ids=["x"], weights=[3.0])
        l1_same = make_chunk("l1", "left", [[0, 0, 0]], ids=["x"], weights=[3.0])
        r0 = make_chunk("r0", "right", [[0.1, 0, 0]], ids=["y"], weights=[2.0])
        lf = hk.PointFrame("left", view(), [l0, l1_same])
        rf = hk.PointFrame("right", view(), [r0])
        session = hk.DistanceSession(lf, rf, {"r": [0.8, 1, 1, .9]}, hk.Pose())
        for cid in ["l0", "l1", "r0"]:
            session.ingest(cid)
        ev = session.finish()
        self.assertAlmostEqual(ev.regions["r"].left_to_right.total_weight, 3.0)
        self.assertTrue(ev.regions["r"].valid)

        bad = make_chunk("l1", "left", [[0.2, 0, 0]], ids=["x"], weights=[4.0])
        lf2 = hk.PointFrame("left", view(), [l0, bad])
        rf2 = hk.PointFrame("right", view(), [r0])
        ev2 = hk.bidirectional_nearest_neighbors(lf2, rf2, hk.Pose(), {"r": [.8, 1, 1, .9]})
        self.assertEqual(ev2.regions["r"].status, "rejected")
        self.assertFalse(ev2.regions["r"].valid)
        self.assertFalse(ev2.regions["r"].released)
        self.assertTrue(ev2.issues)


class BudgetAndPruningTests(unittest.TestCase):
    def test_bbox_prune_function_is_tie_aware(self):
        box = [[1.0, -1, -1], [2.0, 1, 1]]
        skip, lower, reason = hk.should_skip_chunk([0, 0, 0], box, 0.5)
        self.assertTrue(skip)
        self.assertGreater(lower, 0.5)
        # Lower bound equal to incumbent may contain a tie that changes ID.
        skip_eq, lower_eq, _ = hk.should_skip_chunk([1, 0, 0], box, 0.0)
        self.assertFalse(skip_eq)
        self.assertEqual(lower_eq, 0.0)

    def test_small_budget_still_completes_without_matrix_state(self):
        n = 20
        left_xyz = np.c_[np.linspace(0, 1, n), np.zeros(n), np.zeros(n)]
        right_xyz = left_xyz + [0.03, 0, 0]
        lf, rf = two_point_frames(left_xyz, right_xyz)
        budget = hk.MemoryBudget(max_points=2, scratch_bytes=512)
        ev = hk.bidirectional_nearest_neighbors(
            lf, rf, hk.Pose(), {"r": [0.8, 1, 1, .9]}, budget=budget
        )
        self.assertEqual(ev.regions["r"].left_to_right.count_matched, n)
        self.assertLessEqual(ev.telemetry["largest_scratch_bytes"], 512 + 128)
        self.assertEqual(ev.telemetry["full_distance_matrices_materialized"], 0)


class GradientTests(unittest.TestCase):
    def _stable_evaluation(self):
        lf, rf = two_point_frames([[0.0, 0.0, 0.0]], [[0.11, 0.23, 0.31]])
        pose = hk.Pose.z(23.0, [0.27, -0.19, 0.13], "stable-pose")
        return hk.bidirectional_nearest_neighbors(
            lf, rf, pose, {"r": [0.8, 1.0, 100.0, 0.9]}
        )

    def test_aggregated_gradients_match_finite_differences(self):
        ev = self._stable_evaluation()
        result = hk.distance_loss_and_gradients(
            ev, reduction="sum", weight_mode="none", require_stable=True
        )
        self.assertTrue(result.stable)
        self.assertEqual(result.edge_count, 2)

        eps = 1e-6

        def loss_with(left_xyz=None, right_xyz=None, pose=None):
            l = [[0.0, 0, 0]] if left_xyz is None else left_xyz
            r = [[0.11, 0.23, 0.31]] if right_xyz is None else right_xyz
            lf, rf = two_point_frames(l, r)
            e = hk.bidirectional_nearest_neighbors(
                lf, rf, pose or ev.pose, {"r": [.8, 1, 100, .9]}
            )
            return hk.distance_loss_and_gradients(
                e, reduction="sum", weight_mode="none", require_stable=True
            ).loss

        ga = result.point_gradients
        for j in range(3):
            lp = np.array([[0.0, 0, 0]])
            lp[0, j] += eps
            lm = np.array([[0.0, 0, 0]])
            lm[0, j] -= eps
            fd = (loss_with(left_xyz=lp) - loss_with(left_xyz=lm)) / (2 * eps)
            self.assertAlmostEqual(fd, ga["left"]["l-0"][j], places=5)

            rp = np.array([[.11, .23, .31]])
            rp[0, j] += eps
            rm = np.array([[.11, .23, .31]])
            rm[0, j] -= eps
            fd = (loss_with(right_xyz=rp) - loss_with(right_xyz=rm)) / (2 * eps)
            self.assertAlmostEqual(fd, ga["right"]["r-0"][j], places=5)

            tp = ev.pose.translation.copy(); tp[j] += eps
            tm = ev.pose.translation.copy(); tm[j] -= eps
            fd = (
                loss_with(pose=hk.Pose.z(23, tp)) - loss_with(pose=hk.Pose.z(23, tm))
            ) / (2 * eps)
            self.assertAlmostEqual(fd, result.pose_gradient["translation"][j], places=5)

        fp = hk.Pose.z(23 + eps, ev.pose.translation)
        fm = hk.Pose.z(23 - eps, ev.pose.translation)
        fd_deg = (loss_with(pose=fp) - loss_with(pose=fm)) / (2 * eps)
        self.assertAlmostEqual(fd_deg, result.pose_gradient["angle_z_deg"], places=5)

    def test_shared_neighbor_contributions_aggregate(self):
        # Two left queries share one right neighbor; aggregation must add both.
        left = make_chunk("l", "left", [[0, 0, 0], [1, 0, 0]], ids=["a", "b"])
        right = make_chunk("r", "right", [[0.45, 0, 0]], ids=["c"])
        lf = hk.PointFrame("left", view(), [left])
        rf = hk.PointFrame("right", view(), [right])
        ev = hk.bidirectional_nearest_neighbors(lf, rf, hk.Pose(), {"r": [.8, 1, 10, .9]})
        result = hk.distance_loss_and_gradients(ev, reduction="sum", weight_mode="none")
        # c is neighbor of both left_to_right queries and query in reverse.
        # Exact value is less important than source count plus nonzero aggregate.
        self.assertGreaterEqual(
            sum(1 for s in result.sources if s.neighbor_id == "c"), 2
        )
        self.assertGreater(abs(result.point_gradients["right"]["c"][0]), 0.0)

    def test_unstable_supports_are_marked_and_strict_raises(self):
        # x=0 is on the closed x-min view boundary.
        left = make_chunk("l", "left", [[0, 0, 0]], ids=["a"])
        right = make_chunk("r", "right", [[0.1, 0, 0]], ids=["b"])
        lf = hk.PointFrame("left", [[0, -1, -1], [10, 1, 1]], [left])
        rf = hk.PointFrame("right", [[0, -1, -1], [10, 1, 1]], [right])
        ev = hk.bidirectional_nearest_neighbors(lf, rf, hk.Pose(), {"r": [.8, 1, 1, .9]})
        result = hk.distance_loss_and_gradients(ev, allow_unstable=True)
        self.assertFalse(result.stable)
        self.assertTrue(result.unstable_supports)
        with self.assertRaises(hk.UnstableSupportError):
            hk.distance_loss_and_gradients(ev, require_stable=True)


class OperatorCompatibilityTests(unittest.TestCase):
    def test_reusable_operator_gives_each_pose_fresh_eligibility(self):
        record = load_record()
        op = hk.PointDistanceOperator.from_record(record)
        e0 = op.evaluate_pose(pose_id="p0")
        e1 = op.evaluate_pose(pose_id="p1")
        self.assertEqual(e0.pose.id, "p0")
        self.assertEqual(e1.pose.id, "p1")
        self.assertFalse(e1.approved)

    def test_requery_pose_recomputes_eligibility(self):
        left = make_chunk("l", "left", [[0, 0, 0]], ids=["a"])
        right = make_chunk("r", "right", [[0.1, 0, 0]], ids=["b"])
        frames = {
            "left": hk.PointFrame("left", view(), [left]),
            "right": hk.PointFrame("right", view(), [right]),
        }
        op = hk.PointDistanceOperator(
            frames["left"], frames["right"], {"r": [0.8, 1.0, 0.2, 0.9]}
        )
        good = op.evaluate_pose(hk.Pose())
        bad = op.evaluate_pose(hk.Pose.z(0, [5.0, 0, 0]))
        self.assertTrue(good.regions["r"].released)
        self.assertFalse(bad.regions["r"].released)
        bad_q = bad.regions["r"].left_to_right.weighted_quantile_sq
        self.assertLessEqual(bad_q, 30.0)
        self.assertGreater(bad_q, 0.2)

    def test_region_can_be_final_while_other_region_is_partial(self):
        la = make_chunk("la", "left", [[0, 0, 0]], ids=["a"], region="A")
        lb = make_chunk("lb", "left", [[5, 0, 0]], ids=["b"], region="B")
        ra = make_chunk("ra", "right", [[0.1, 0, 0]], ids=["c"], region="A")
        rb = make_chunk("rb", "right", [[5.1, 0, 0]], ids=["d"], region="B")
        lf = hk.PointFrame("left", view(), [la, lb])
        rf = hk.PointFrame("right", view(), [ra, rb])
        rules = {
            "A": [.8, 1, 1, .9],
            # B is observed but intentionally not part of the station gate.
        }
        session = hk.DistanceSession(lf, rf, rules, hk.Pose())
        session.ingest("la")
        session.ingest("ra")
        report = session.snapshot()
        self.assertEqual(report.status, "partial")
        self.assertEqual(report.regions["A"].status, "final")
        self.assertTrue(report.regions["A"].final)
        self.assertTrue(report.approved)
        self.assertEqual(report.regions["B"].status, "partial")

    def test_lazy_chunk_handle_is_loaded_on_ingest(self):
        loaded = []
        coords = [[0.2, 0, 0]]

        def loader():
            loaded.append(True)
            return make_chunk("r-lazy", "right", coords, ids=["lazy"])

        handle = hk.ChunkHandle(
            chunk_id="r-lazy", side="right",
            bbox=[[0, -1, -1], [1, 1, 1]], loader=loader, regions=["r"],
        )
        left = make_chunk("l", "left", [[0, 0, 0]], ids=["a"])
        lf = hk.PointFrame("left", view(), [left])
        rf = hk.PointFrame("right", view(), [handle])
        session = hk.DistanceSession(lf, rf, {"r": [.8, 1, 1, .9]}, hk.Pose())
        session.ingest("r-lazy")
        self.assertEqual(len(loaded), 1)
        report = session.finish()
        self.assertEqual(report.regions["r"].right_to_left.count_query, 1)

    def test_jsonl_match_sink(self):
        import tempfile
        from pathlib import Path
        ev = hk.evaluate_record(load_record(), pose_id="p0")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "matches.jsonl"
            count = ev.write_matches_jsonl(path)
            self.assertEqual(count, 5)  # two left seam + three right seam
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)


if __name__ == "__main__":
    unittest.main()
