{\rtf1\ansi\ansicpg1251\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;\f1\fnil\fcharset0 HelveticaNeue-Italic;}
{\colortbl;\red255\green255\blue255;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cspthree\c100000\c100000\c100000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\fs26 \cf2 for i, (input_data, target) in enumerate(zip(test_inputs, test_targets)):\
            prediction = self.predict(input_data)\
            predicted_class = 1 if prediction[0] > 0.5 else 0\
            actual_class = target[0]\
            \
            status = "\uc0\u1042 \u1045 \u1056 \u1053 \u1054 " if predicted_class == actual_class else "\u1054 \u1064 \u1048 \u1041 \u1050 \u1040 "\
            print(f"\uc0\u1055 \u1088 \u1080 \u1084 \u1077 \u1088  \{i+1\}: \u1042 \u1093 \u1086 \u1076  \{input_data\} -> \u1055 \u1088 \u1077 \u1076 \u1089 \u1082 \u1072 \u1079 \u1072 \u1085 \u1080 \u1077 : \{prediction[0]:.4f\} (\{predicted_class\}) | \u1054 \u1078 \u1080 \u1076 \u1072 \u1085 \u1080 \u1077 : \{actual_class\} \{status\}")\
            \
            if predicted_class == actual_class:\
                correct += 1\
        \
        accuracy = correct / total * 100\
        print(f"\uc0\u1048 \u1090 \u1086 \u1075 \u1086 \u1074 \u1072 \u1103  \u1090 \u1086 \u1095 \u1085 \u1086 \u1089 \u1090 \u1100 : \{accuracy:.1f\}% (\{correct\}/\{total\})")\
        return accuracy\
\
def demo_xor():\
    print("=" * 50)\
    print("\uc0\u1044 \u1045 \u1052 \u1054  \u1053 \u1045 \u1049 \u1056 \u1054 \u1057 \u1045 \u1058 \u1048 : \u1047 \u1040 \u1044 \u1040 \u1063 \u1040  XOR")\
    print("=" * 50)\
    \
    training_inputs = [\
        [0, 0],\
        [0, 1],\
        [1, 0],\
        [1, 1]\
    ]\
    \
    training_targets = [\
        [0],\
        [1],\
        [1],\
        [0]\
    ]\
    \
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)\
    nn.train(training_inputs, training_targets, epochs=10000)\
    nn.test(training_inputs, training_targets)\
\
def demo_and():\
    print("\\n" + "=" * 50)\
    print("\uc0\u1044 \u1045 \u1052 \u1054  \u1053 \u1045 \u1049 \u1056 \u1054 \u1057 \u1045 \u1058 \u1048 : \u1047 \u1040 \u1044 \u1040 \u1063 \u1040  AND")\
    print("=" * 50)\
    \
    training_inputs = [\
        [0, 0],\
        [0, 1],\
        [1, 0],\
        [1, 1]\
    ]\
    \
    training_targets = [\
        [0],\
        [0],\
        [0],\
        [1]\
    ]\
    \
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.5)\
    nn.train(training_inputs, training_targets, epochs=8000)\
    nn.test(training_inputs, training_targets)\
\
def demo_or():\
    print("\\n" + "=" * 50)\
    print("\uc0\u1044 \u1045 \u1052 \u1054  \u1053 \u1045 \u1049 \u1056 \u1054 \u1057 \u1045 \u1058 \u1048 : \u1047 \u1040 \u1044 \u1040 \u1063 \u1040  OR")\
    print("=" * 50)\
    \
    training_inputs = [\
        [0, 0],\
        [0, 1],\
        [1, 0],\
        [1, 1]\
    ]\
    \
    training_targets = [\
        [0],\
        [1],\
        [1],\
        [1]\
    ]\
    \
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)\
    nn.train(training_inputs, training_targets, epochs=6000)\
    nn.test(training_inputs, training_targets)\
\
def custom_problem():\
    print("\\n" + "=" * 50)\
    print("\uc0\u1055 \u1054 \u1051 \u1068 \u1047 \u1054 \u1042 \u1040 \u1058 \u1045 \u1051 \u1068 \u1057 \u1050 \u1040 \u1071  \u1047 \u1040 \u1044 \u1040 \u1063 \u1040 ")\
    print("=" * 50)\
    \
    training_inputs = [\
        [0.1, 0.2],\
        [0.3, 0.4],\
        [0.6, 0.7],\
        [0.8, 0.9]\
    ]\
    \
    training_targets = [\
        [0.1],\
        [0.3],\
        [0.6],\
        [0.8]\
    ]\
    \
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=1, learning_rate=0.3)\
    nn.train(training_inputs, training_targets, epochs=5000)\
    \
    print("\uc0\u1058 \u1077 \u1089 \u1090 \u1080 \u1088 \u1086 \u1074 \u1072 \u1085 \u1080 \u1077  \u1085 \u1072  \u1085 \u1086 \u1074 \u1099 \u1093  \u1076 \u1072 \u1085 \u1085 \u1099 \u1093 :")\
    test_inputs = [[0.2, 0.3], [0.5, 0.6], [0.7, 0.8]]\
    for i, inp in enumerate(test_inputs):\
        prediction = nn.predict(inp)\
        print(f"\uc0\u1042 \u1093 \u1086 \u1076 : \{inp\} -> \u1055 \u1088 \u1077 \u1076 \u1089 \u1082 \u1072 \u1079 \u1072 \u1085 \u1080 \u1077 : \{prediction[0]:.4f\}")\
\
if
\f1\i  name 
\f0\i0 == "__main__":\
    print("\uc0\u1055 \u1056 \u1054 \u1057 \u1058 \u1040 \u1071  \u1053 \u1045 \u1049 \u1056 \u1054 \u1057 \u1045 \u1058 \u1068  \u1053 \u1040  PYTHON")\
    print()\
    \
    demo_xor()\
    demo_and()\
    demo_or()\
    custom_problem()\
    \
    print("\\n" + "=" * 50)\
    print("\uc0\u1042 \u1057 \u1045  \u1058 \u1045 \u1057 \u1058 \u1067  \u1047 \u1040 \u1042 \u1045 \u1056 \u1064 \u1045 \u1053 \u1067 ")\
    print("=" * 50)}