"""
One-command demo: fine-tune MethylBERT for 60 steps on a 780-read subset of the
Excitable family (Heart-Cardio, Neuron, Oligodend), then evaluate.

    python demo/run_demo.py

Downloads the pretrained hanyangii/methylbert_hg19_2l checkpoint from Hugging Face
on first use. Outputs (metrics, confusion matrices, curves) go to demo/output/.
The demo checks that the pipeline runs; 60 steps on 600 reads is not a result.
"""
import multiprocessing
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # The dataset loader uses a multiprocessing pool, and run_training.py has no
    # main guard, so the "spawn" default on macOS would re-run the whole script in
    # every worker. "fork" (the Linux default, used for the original runs) avoids that.
    if sys.platform != "win32":
        multiprocessing.set_start_method("fork", force=True)
    os.chdir(REPO)
    os.makedirs(os.path.join("demo", "output"), exist_ok=True)  # trainer mkdirs only the leaf folder
    sys.argv = ["run_training.py", "-c", os.path.join("demo", "demo_config.json")]
    runpy.run_path(os.path.join("Training", "run_training.py"), run_name="__main__")
