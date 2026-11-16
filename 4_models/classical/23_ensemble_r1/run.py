"""23_ensemble_r1: ported from the teammate branch (its REPORT.md section 12) through code/classic_adapter.py."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main
from classic_adapter import harness_predict

predict = harness_predict('ensemble_r1')


if __name__ == '__main__':
    main(predict, __file__)
