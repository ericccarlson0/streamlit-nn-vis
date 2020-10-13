import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import glob
import os
import torch
import torchvision

BASE_DIR = ""
TRAINED_MODEL_DIR = ""

trained_model_names = [fname[len(TRAINED_MODEL_DIR) + 1:]
                       for fname in glob.iglob(TRAINED_MODEL_DIR + "**/*.pt", recursive=False)]
trained_model_names.sort()


def main():
    st.title("Model Visualization")
    trained_model = st.selectbox("Select a model to visualize:", trained_model_names)
    st.subheader(f"You chose {trained_model}. It does not matter, though, because the only model we are "
                 f"considering for now is the pre-trained ShuffleNet model from torchvision.")

    model = load_model()

    radius = st.sidebar.selectbox("Radius of Histogram:", [2, 4, 8, 16])
    divisions = st.sidebar.selectbox("# Buckets in Histogram:", [8, 16, 32, 64])
    buckets = [0] * (divisions + 1)
    named_parameters = list(model.named_parameters())

    if st.sidebar.checkbox("Do you want to show a range of layers?"):
        layer_nums = st.sidebar.slider("Choose the layer numbers you want to include (higher -> deeper).",
                                       0, 170, [0, 16])

        for i in range(layer_nums[0], layer_nums[1]):
            new_buckets = get_layer_buckets(model=model, layer=i, radius=radius, divisions=divisions)
            buckets = [
                buckets[j] + new_buckets[j]
                for j in range(divisions + 1)
            ]

    else:
        layer_num = st.sidebar.slider("Choose the layer number you want to include (higher -> deeper).",
                                      0, 178, 0)

        buckets = get_layer_buckets(model=model, layer=layer_num, radius=radius, divisions=divisions)
        st.sidebar.markdown(f"""This set of parameters is called `{named_parameters[layer_num][0]}`.""")

    buckets_df = pd.DataFrame({
        "Bucket": [i * (2.0 * radius / divisions) for i in range(divisions + 1)],
        "# Parameters": buckets
    })
    st.altair_chart(alt.Chart(buckets_df).mark_bar().encode(
        x="Bucket",
        y="# Parameters",
        color=alt.value("orange")
    ))

    evaluate_near_zero(radius=radius, divisions=divisions, buckets=buckets, n=1)
    evaluate_near_zero(radius=radius, divisions=divisions, buckets=buckets, n=3)


"""
This class is used to generate a bucketing function which can then be used to generate bucket indices from inputs. 
Matplotlib, Altair, and essentially all visualization libraries make it so that something like this never has to be 
written. This is mainly here because I wanted to turn it into a ufunc to see if it actually worked.
"""
class Bucketer:
    def __init__(self, radius: int, divisions: int):
        self.radius = radius
        self.divisions = divisions

    def __call__(self, value: int) -> int:
        if value > self.radius:
            return self.divisions
        elif value < -self.radius:
            return 0

        return int((value * self.divisions / (2 * self.radius))
                   + (self.divisions / 2)
                   + 0.5)

"""
An element that is used to evaluate the amount of parameters near zero. This is important if you want to deal with 
issues such as dead ReLU or Internal Covariate Shift.
"""
def evaluate_near_zero(radius: int, divisions: int, buckets: list, n: int):
    # If there are N+1 buckets, the center bucket will be indexed by (N+1)/2 - 1 = N/2 - 1/2.
    # This means that we should take the floor function.
    start_idx = (divisions - n)//2

    near_zero = 0
    for i in range(start_idx, start_idx + n):
        near_zero += buckets[i]

    bound = n * radius / divisions

    total_params = sum(buckets)
    st.markdown(f"**{100 * near_zero / total_params: .2f}%** of params are in (-{bound}, {bound})")


"""
Generates the model. The fact that this is cached means that the cache should be cleared when the model is changed.
"""
@st.cache
def load_model():
    return torchvision.models.shufflenet_v2_x0_5(pretrained=True)

"""
Generates a list representing the number of parameters in each bucket, where the parameters come from the specified
specified layer of the specified model. The buckets represent the range from -radius to radius, with the specified
number of divisions. Those values outside this range are put in the outermost buckets.
"""
@st.cache
def get_layer_buckets(model: torch.nn.Module, layer: int, radius: int, divisions: int):
    bucketer = Bucketer(radius=radius, divisions=divisions)
    ufunc = np.frompyfunc(bucketer, nin=1, nout=1)
    buckets = [0] * (divisions + 1)

    param_tensor = list(model.parameters())[layer]
    param_array = param_tensor.detach().numpy()
    bucketed_params = ufunc(param_array)

    for n in bucketed_params.flatten(order='K'):
        buckets[n] += 1

    # st.warning(f"Buckets (layer {layer}): {buckets}")
    return buckets

# Enter the app through the main loop.
if __name__ == '__main__':
    main()
