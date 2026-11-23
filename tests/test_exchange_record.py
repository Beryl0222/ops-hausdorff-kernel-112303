"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from hausdorff_kernel_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(record["arrival"][-1] == "r/late" and len(record["point_sets"]["right"][-1]["points"]) > 1)
        self.assertTrue(record["point_sets"]["left"][0]["points"][1] == record["point_sets"]["left"][1]["points"][0])
        self.assertTrue(np.dot(record["point_sets"]["left"][0]["points"][2][3], record["point_sets"]["right"][0]["points"][2][3]) == 0)
        self.assertTrue(len({p["angle_z_deg"] for p in record["pose_right_to_left"]}) == 2)
