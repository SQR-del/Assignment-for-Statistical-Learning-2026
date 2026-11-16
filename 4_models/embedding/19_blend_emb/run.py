"""19_blend_emb: ported from the teammate branch (its methods/13_final_blend) through code/emb_adapter.py."""
import os, sys
_R = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_R, "src")): _R = os.path.dirname(_R)
sys.path[:0] = [os.path.join(_R, "src"), os.path.join(_R, "3_validation")]
from harness import main
from emb_adapter import harness_predict

predict = harness_predict('blend_b')


if __name__ == '__main__':
    main(predict, __file__)
