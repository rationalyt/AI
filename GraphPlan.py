import sys

#Reading the file for processing

args = sys.argv[1]
f = open(args,"r")
k = f.read().split("\n")

#Creating a list of dictionaries consisting of actions as keys
#Precondition dictionary -> precond()
#Effect dictionary -> effect()
#init_goal -> consists of initial state and goal state

precond = dict()
effect = dict()
init_goal = dict()
action = []
action_list = []
g = 0
i = 0

#By identifying spaces,the actions are seperated and the corresponding dictionaries are updated
while i < len(k):
    if k[i] != '':
        a = k[i].split(": ")
        if a[0] == 'ACTION':
            a[1] = a[1][1:]
            action.append(a[1])
            i = i+1
            precond[a[1]] = k[i].split(": ")[1].split(",")
            i = i+1
            effect[a[1]] = k[i].split(":  ")[1].split(",")
        else:
            init_goal[a[0]] = a[1].split(",")
    i = i+1

traversal_stack = []
new_state_list = init_goal['INIT']
traversal_stack.append(new_state_list)
print(f"The intial state is {init_goal['INIT']}")
print(f"The set of actions are: {action}")
print(f"The goal state is {init_goal['GOAL']}")
new_state = []
flag = 0
print("\n\n")


#Returns the negated string of p
def negated(p):
    if p[0] == '-':
        return p[1:]
    else:
        return '-'+p

#Checks for interference actions
def check_interference_actions(action_list,new_state):
    m2 = []
    for a in action_list:
        m2 = []
        for i in effect[a]:
            m2 = []
            if (i[0] == '-'):
                for j in action_list:
                    if i[1:] in precond[j]:
                        m2.append(j)
                if m2 != []:
                    if a in m2:
                      m2.remove(a)
                    print(f"{m2} and {a} is interference mutex relationship")
            elif (i[0] != '-'):
                ns = '-'+i
                for j in action_list:
                    if ns in precond[j]:
                        m2.append(j)
                if m2 != []:
                    if len(m2)==1 and m2[0]!=a:
                        print(f"{m2} and {a} is interference mutex relationship")
                    elif len(m2) > 1 and (a in m2):
                        m2.remove(a)
                        print(f"{m2} and {a} is interference mutex relationship")


#Checks for inconsistent actions
def check_inconsistent_actions(action_list,new_state):
    for s in new_state:
        m1, m2 = [], []
        if (s[0] == '-') and (s[1:] in new_state):
            for a in action_list:
                if s in effect[a]:
                    m1.append(a)
                if s[1:] in effect[a]:
                    m2.append(a)
            if m2 == []:
                print(f"{m1} and persistent action from {s[1:]} is inconsistent and interference mutex relationship")
            elif m1==[]:
                print(f"{m2} and persistent action from {s} is inconsistent and interference mutex relationship")
            else:
                print(f"{m1} and {m2} actions are in inconsistent mutex relationship")

#Check for competing needs
def check_competing_needs(action_list,new_state):
    competing_set = set()
    for a in action_list:
        for p in precond[a]:
            negated_p = negated(p)
            for b in action_list:
                if negated_p in precond[b]:
                    competing_set.add((a,b))
    for (x,y) in competing_set:
        #competing_set.remove((y,x))
        print(f"{x} is a competing action of {y}")

#Check for mutex literals
def check_mutex_literals(new_state):
    for i in new_state:
        if (i[0] == '-') and (i[1:] in new_state):
            print(f"{i} and {i[1:]} are in Negated Literals relationship")
    print("\n\n")

#Check if goal literals are present
def check_goal(new_state):

    for i in init_goal['GOAL']:
        if i not in new_state:
            return -1
    return 1

#Execute action a to generate new layer
def execute(a,new_state):
    for i in effect[a]:
        if i not in new_state:
           new_state.append(i)

#Keeps generating layers till goal states repeats
while True:
    new_state = new_state_list.copy()
    for a in action:
        flag = 0
        for p in precond[a]:
            if p not in new_state_list:
                flag = -1
                break
        if flag == 0:
           action_list.append(a)
           execute(a,new_state)  #Execute action a on S(i-1) to generate S(i+1)
    print(f"Generating Action Layer: {action_list}")

    # Check mutex actions
    print("Mutex Arcs:")
    check_inconsistent_actions(action_list.copy(),new_state.copy())
    check_interference_actions(action_list.copy(), new_state.copy())
    check_competing_needs(action_list.copy(), new_state.copy())
    print("\n\n")
    print(f"Generating State Layer: {new_state}")
    print("Mutex Arcs:")
    #Check mutex literals
    check_mutex_literals(new_state)
    traversal_stack.append(action_list.copy())
    traversal_stack.append(new_state)
    if check_goal(new_state) == 1:
        g = g + 1
        if g == 3:
         print("\nThe traversal stack is:")
         #Prints all layers generated
         for i in traversal_stack:
          print(i)
         break
    action_list.clear()
    new_state_list = new_state.copy()












