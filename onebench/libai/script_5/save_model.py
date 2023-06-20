"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
import oneflow.nn as nn
import flowvision



class MyGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, *input):
        return self.model(*input)


if __name__ == "__main__":
    image1 = flow.ones((1, 3, 224, 224))
    model = flowvision.models.style_transfer.fast_neural_style(pretrained=True, progress=True, style_model = "sketch")
    model.eval()
    graph = MyGraph(model)
    out = graph(image1)

    flow.save(graph, "/data/home/sunjinfeng/torch_mock/1/model")