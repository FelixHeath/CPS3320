class BinaryTreeNode:
    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None


def insert(root, newValue):
    # if binary search tree is empty, make a new node and declare it as root
    if root is None:
        root = BinaryTreeNode(newValue)
        return root
    # binary search tree is not empty, so we will insert it into the tree
    # if newValue is less than value of data in root, add it to left subtree and proceed recursively
    if newValue < root.data:
        root.leftChild = insert(root.leftChild, newValue)
    else:
        # if newValue is greater than value of data in root, add it to right subtree and proceed recursively
        root.rightChild = insert(root.rightChild, newValue)
    return root


def search(root, value,counter):
    # Condition 1
    if root == None:
        print(False)
    # Condition 2
    elif root.data == value:
        print(True)
        print("total steps are: ", counter)
    # Condition 3
    elif root.data < value:
        counter = counter + 1;
        return search(root.rightChild, value, counter)

    # Condition 4
    elif root.data > value:
        counter = counter + 1;
        return search(root.leftChild, value,counter)

    else:
        print(False)


def inorder(root):
    if root:
        inorder(root.leftChild)
        print(root.data)
        inorder(root.rightChild)


root = insert(None, 25)
insert(root, 19)
insert(root, 28)
insert(root, 13)
insert(root, 9)
insert(root, 30)
insert(root, 17)
insert(root,26)

search(root, 17,1)
print("")
search(root, 50,1)

print("    ")
inorder(root)