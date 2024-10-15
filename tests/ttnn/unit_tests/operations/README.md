Low PCC issue for the following test case ttnn.reshape is failing in lenet model.

To recreate the issue run the command:
`pytest tests/ttnn/unit_tests/operations/test_layout.py::test_layout`

For input shape [1, 1, 28, 28]

This code implements a part of LeNet model and includes testing utilities. A notable issue is the drop in PCC when changing the tensor layout while the tensor is on the device.

E       AssertionError: 0.03920856780939073
