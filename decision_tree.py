import sys

#Opening file Problem.txt
f = open('Problem.txt', "r")

choice, lottery, outcome = {}, {}, {}
l, c, flag = 0, 0, 0
statements = []

#Reading the file and extracting lottery,choice,outcome
for line in f.readlines():

    if 'CHOICE:' in line:
        flag = 1
        k = line.split(" ")[1]
        c = k[0:-1]
        choice[c] = []

    elif ('ENDCHOICE' in line) or ('ENDLOTTERY' in line):
        flag = 0

    elif 'LOTTERY:' in line:
        flag = 2
        k = line.split(" ")[1]
        l = k[0:-1]
        lottery[l] = []

    elif 'OUTCOME:' in line:
        k = line.split(" ")
        outcome[k[1]] = k[2][0:-1]
    
    elif line != '\n':
        if flag == 1:
            k = line.split(" ")
            choice[c].append((k[1], k[2][0:-1]))
        if flag == 2:
            k = line.split(" ")
            lottery[l].append((k[1], k[2], k[3][0:-1]))

#Dictionary of objects
object_dict = {}

#Class used to recursively generate the tree
class Node:

    counter = 0

    def __init__(self, name, parent):

        self.name = name
        self.number = 0
        self.value = -1
        self.arrows = []
        self.values = []
        self.children = []
        self.type = ""
        self.parent = parent
        self.decision = ""

        if name in choice.keys():

            self.type = "choice"
            Node.counter = Node.counter + 1
            self.number = Node.counter
            print(f"Adding Node {Node.counter} {self.type} {self.name} {self.parent}\n")
            for e in choice[name]:
                self.arrows.append(e[0])
                self.children.append(Node(e[1], name))
            max = -sys.maxsize
            index = 0
            for e in range(0, len(self.children)):
                if float(self.children[e].value) > max:
                    index = e
                    max = float(self.children[e].value)
            self.value = self.children[index].value
            self.decision = self.arrows[index]


        if name in lottery.keys():

            self.type = "lottery"
            Node.counter = Node.counter + 1
            self.number = Node.counter
            print(f"Adding Node {Node.counter} {self.type} {self.name} {self.parent}\n")
            for e in lottery[name]:
                self.arrows.append(e[0])
                self.values.append(float(e[1]))
                self.children.append(Node(e[2], name))
            for e in range(0, len(self.children)):
                self.value = self.value + (self.values[e] * self.children[e].value)

        if name in outcome.keys():

            self.type = "outcome"
            Node.counter = Node.counter + 1
            self.number = Node.counter
            print(f"Adding Node {Node.counter} {self.type} {self.name} {self.parent}\n")
            self.number = Node.counter
            self.value = float(outcome[name])

        object_dict[self.number] = self

# The tree is initialised with the start node
start = Node('Start', 'Start')
print("\n")
for i in range(1, len(object_dict)+1):
    node = object_dict[i]
    if node.type == "choice":
        print(f"Decision Node: {node.number}, {node.name}  {node.decision}  {node.value}\n")
    else:
        print(f"Expected Value Node: {node.number}, {node.name}  {node.value}\n")












