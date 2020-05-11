import tensorflow as tf
def transfer_weights(net, ck_path):
    for layer in net.layers:
        ck = tf.compat.v1.train.NewCheckpointReader(ck_path)

        if layer.weights:
            weights = []

            for w in layer.weights:
                weight_name = w.name.replace(':0', '')
                weight_name = weight_name.split("/")
                x = weight_name[0].split("_")
                print(weight_name)
                name = ""
                gr = int(x[1][0])
                if gr > 2 and gr < 6:
                    name = "InceptionV1/Mixed_" + x[1] + "/Branch_" + x[2][0] + "/Conv2d_0" + x[2][1] + "_" + x[3]
                    if x[1] in "5b" and x[2] in "2b" and x[3] in "3x3":
                        name = "InceptionV1/Mixed_" + x[1] + "/Branch_" + x[2][0] + "/Conv2d_0b_" + x[3]
                    if "conv" in x[4]:
                        if weight_name[1] == 't':
                            name = name + "/temporal"
                            if weight_name[2] == "kernel":
                                name = name + "/model_with_weights"
                            else:
                                name = name + "/biases"
                        elif weight_name[1] == "self_gating":
                            name = name + "/self_gating/transformer_W/model_with_weights"
                        else:
                            name = name + "/model_with_weights"
                        weights.append(ck.get_tensor(name))
                    elif "bn" in x[4]:
                        name = name + "/BatchNorm/" + weight_name[1]
                        t = ck.get_tensor(name)
                        # print(t.shape[-1])
                        weights.append(t.reshape(t.shape[-1]))

                #    print(name)
                elif gr == 6:
                    name = "InceptionV1/Logits/Conv2d_0c_1x1"
                    if weight_name[1] in "kernel":
                        name = name + "/model_with_weights"
                    else:
                        name = name + "/biases"
                    weights.append(ck.get_tensor(name))
                else:
                    name = "InceptionV1/Conv2d_" + x[1] + "_" + x[2]
                    if "conv" in x[3]:
                        if weight_name[1] == 't':
                            name = name + "/temporal"
                            if weight_name[2] == "kernel":
                                name = name + "/model_with_weights"
                            else:
                                name = name + "/biases"
                        elif weight_name[1] == "self_gating":
                            name = name + "/self_gating/transformer_W/model_with_weights"
                        else:
                            name = name + "/model_with_weights"
                        weights.append(ck.get_tensor(name))
                    elif "bn" in x[3]:
                        print(weight_name[1])
                        name = name + "/BatchNorm/" + weight_name[1]
                        t = ck.get_tensor(name)
                        # print(t.shape[-1])
                        weights.append(t.reshape(t.shape[-1]))
                #    print("bn")
                print(name)

            layer.set_weights(weights)
    return net