import oneflow as flow

m = flow.nn.Linear(2,3)
print(m.state_dict())

myparams = {"weight":flow.ones(3,2), "bias":flow.zeros(3)}
m.load_state_dict(myparams)

print(m.state_dict())

flow.save(m.state_dict(), "./model")

params = flow.load("./model")
m2 = flow.nn.Linear(2,3)
m2.load_state_dict(params)

print(m2.state_dict())