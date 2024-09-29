from typing import Any, Tuple
import customtkinter as ctk
from CTkScrollableDropdown import CTkScrollableDropdown

class ArchitectureFilterFrame(ctk.CTkFrame):
    '''Frame that contains the architecture filter widgets, there are three checkboxes for the architectures: 'fullyconnected', 'convolutional', 'residual' '''
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        # Configure the frame to handle 3 columns with equal weight and 2 rows with equal weight
        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure((0, 1), weight=1)
        self.isFullyConnectedIncluded = ctk.BooleanVar(value=True)
        self.isConvolutionalIncluded = ctk.BooleanVar(value=True)
        self.isResidualIncluded = ctk.BooleanVar(value=True)
        self.createWidgets()

    def checkInteraction(self, *params):
        # If all the checkboxes are unchecked, check them all (to match the logic.py behavior)
        included_architectures = []
        if self.isFullyConnectedIncluded.get():
            included_architectures.append('fullyconnected')
        if self.isConvolutionalIncluded.get():
            included_architectures.append('convolutional')
        if self.isResidualIncluded.get():
            included_architectures.append('residual')

        self.winfo_toplevel().logic_instance.included_architectures = included_architectures
        self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()
        

    def createWidgets(self):
        self.architectureLabel = ctk.CTkLabel(self, text="Architectures:", font=("Arial", 14, "bold"))
        self.architectureLabel.grid(row=0, column=0, columnspan = 3, pady=2, padx=2)
        # Checkbox widget for fully connected architecture:
        self.FullyConnectedCallbackName = self.isFullyConnectedIncluded.trace_add('write', self.checkInteraction)
        self.fullyConnectedCheckBox = ctk.CTkCheckBox(self, text="Fully Connected", variable=self.isFullyConnectedIncluded, font=("Arial", 11, "bold"))
        self.fullyConnectedCheckBox.grid(row=1, column=0, pady=(0,5), padx = 4, sticky="w")
        # Checkbox widget for convolutional architecture:
        self.ConvolutionalCallbackName = self.isConvolutionalIncluded.trace_add('write', self.checkInteraction)
        self.convolutionalCheckbox = ctk.CTkCheckBox(self, text="Convolutional", variable=self.isConvolutionalIncluded, font=("Arial", 11, "bold"))
        self.convolutionalCheckbox.grid(row=1, column=1, pady=(0,5), padx = 4, sticky="w")
        # Checkbox widget for residual architecture:
        self.ResidualCallbackName = self.isResidualIncluded.trace_add('write', self.checkInteraction)
        self.residualCheckbox = ctk.CTkCheckBox(self, text="Residual", variable=self.isResidualIncluded, font=("Arial", 11, "bold"))
        self.residualCheckbox.grid(row=1, column=2, pady=(0,5), padx = 4, sticky="w")

class ParamsFilterFrame(ctk.CTkFrame):
    '''Frame that contains the parameters filter widgets, there are two entry widgets for the minimum and maximum number of parameters, and a button to apply the filter'''
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        # Configure the frame to handle 4 columns with equal weight and 2 rows with equal weight
        self.configure(width=self.winfo_toplevel().winfo_width())
        self.grid_columnconfigure((0, 1, 2, 3), weight=2)
        self.grid_rowconfigure((0, 1), weight=1)
        # Variables to store the values of the entry widgets
        self.minParams = ctk.StringVar(value='0')
        self.maxParams = ctk.StringVar(value=f'{int(1e10)}')
        self.after_id = None
        self.createWidgets()

    def helpAplly(self):
        try:
            old_min = self.winfo_toplevel().logic_instance.min_params
            old_max = self.winfo_toplevel().logic_instance.max_params
            new_min = int(self.minParams.get())
            new_max = int(self.maxParams.get())
            if new_min != old_min or new_max != old_max:
                print(f"Applying params filter button clicked min: {self.minParams.get()} max: {self.maxParams.get()}")
                self.winfo_toplevel().logic_instance.min_params = int(self.minParams.get())
                self.winfo_toplevel().logic_instance.max_params = int(self.maxParams.get())
                self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()
            else:
                print("No changes in the params filter")
        except Exception as e:
            print(f"Error in the input: {e}")

    def applyParamsFilter(self, *params):
        '''Method that is called when the variables minParams or maxParams are modified'''
        # TODO: Make it execute only after a little delay, to avoid multiple calls when the user is typing
        if self.after_id:
            self.winfo_toplevel().after_cancel(self.after_id)
        self.after_id = self.winfo_toplevel().after(1000, self.helpAplly)
        

    def createWidgets(self):
        self.paramsLabel = ctk.CTkLabel(self, text="Number of parameters:", font=("Arial", 14, "bold"))
        self.paramsLabel.grid(row=0, column=0, columnspan = 4, pady=2, padx=2, sticky="nsew")
        # Minimum number of parameters widgets:
        #self.minParams.trace_add('write', self.applyParamsFilter)
        self.minParamsLabel = ctk.CTkLabel(self, text="Min:")
        self.minParamsLabel.grid(row=1, column=0, pady=2, padx=2, sticky="nsew")
        self.minParamsEntry = ctk.CTkEntry(self, width=50, textvariable=self.minParams)
        self.minParamsEntry.bind('<Key>',self.applyParamsFilter)
        self.minParamsEntry.grid(row=1, column=1, pady=2, padx=5, sticky="nsew")
        # Maximum number of parameters widgets:
        #self.maxParams.trace_add('write', self.applyParamsFilter)
        self.maxParamsLabel = ctk.CTkLabel(self, text="Max:")
        self.maxParamsLabel.grid(row=1, column=2, pady=2, padx=2, sticky="nsew")
        self.maxParamsEntry = ctk.CTkEntry(self, width=50, textvariable=self.maxParams)
        self.maxParamsEntry.bind('<Key>',self.applyParamsFilter)
        self.maxParamsEntry.grid(row=1, column=3, pady=2, padx=5, sticky="nsew")

class ListNodesScrollFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, is_include:bool, possible_nodes:list, *args, **kwargs):
        ctk.CTkScrollableFrame.__init__(self, master, *args, **kwargs, height=30, orientation="horizontal")
        self.nodeInList = []
        self.is_include = is_include
        self.createWidgets()
        self.possible_nodes = possible_nodes

    def createWidgets(self):
        self.listNodes = ctk.CTkLabel(self, text=", ".join(self.nodeInList))
        self.listNodes.pack(fill="x")

    def updateList(self):
        self.listNodes.configure(text=", ".join(sorted(set(self.nodeInList))))
        if self.is_include:
            self.winfo_toplevel().logic_instance.included_nodes = sorted(self.nodeInList)
        else:
            self.winfo_toplevel().logic_instance.excluded_nodes = sorted(self.nodeInList)

        self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()
        print(f"Updated node list: {self.nodeInList}")

    def addNode(self, node):
        if node not in self.possible_nodes:
            print(f"Node {node} not in the list of nodes, not added")
            return
        if node not in self.nodeInList:
            self.nodeInList.append(node)
            self.updateList()
        else:
            print(f"Node {node} already in the list, not added")
        
    def removeNode(self, node):
        try:
            self.nodeInList.remove(node)
            self.updateList()
        except:
            print(f"Node {node} not in the list, cannot remove")
        
class AddNodesFilterFrame(ctk.CTkFrame):
    """Frame that contains the nodes add filter widgets, for the add there is a combobox with all the node types and a button to add the selected node. For the remove there is another combobox with all the selected nodes and a button to remove the selected node"""
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        # Configure the frame to handle 3 columns with equal weight and 5 rows with equal weight
        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.exclusionVar = ctk.BooleanVar(value=False)
        self.exclusionCheckbox = None
        self.createWidgets()

    def exclusionCheckboxInteraction(self, *params):
        #print(f"Exclusion checkbox changed to {self.exclusionVar.get()}")
        if self.exclusionVar.get():
            self.winfo_toplevel().logic_instance.excluded_nodes.append("all")
        else:
            while "all" in self.winfo_toplevel().logic_instance.excluded_nodes:
                self.winfo_toplevel().logic_instance.excluded_nodes.remove("all")
        #print(f"Excluded nodes: {self.winfo_toplevel().logic_instance.excluded_nodes}")
        self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()

    def insert_method(self, e, entry):
        entry.delete(0, 'end')
        entry.insert(0, e)

    def createWidgets(self):
        self.nodesLabel = ctk.CTkLabel(self, text="Included Node types:", font=("Arial", 14, "bold"))
        self.nodesLabel.grid(row=0, column=0, columnspan=2, pady=(10, 2), padx=2)
        self.exclusionVar.trace_add('write', self.exclusionCheckboxInteraction)
        self.exclusionCheckbox = ctk.CTkCheckBox(self, text="Exclude others", font=("Arial", 11, "bold"), variable=self.exclusionVar)
        self.exclusionCheckbox.grid(row=0, column=2, pady=(10, 2), padx=2)
        # Add node widgets:
        self.addNodeLabel = ctk.CTkLabel(self, text="Add:")
        self.addNodeLabel.grid(row=1, column=0, pady=(2,5), padx=2)
        # Add node entry using the CTkScrollableDropdown to autocomplete the node types 
        self.addNodeEntry = ctk.CTkEntry(self)
        self.addNodeEntry.grid(row=1, column=1, pady=(0,5), padx=2)
        CTkScrollableDropdown(attach=self.addNodeEntry, values=self.winfo_toplevel().logic_instance.all_nodes, command=lambda e: self.insert_method(e, self.addNodeEntry), autocomplete=True) 
        self.addNodeButton = ctk.CTkButton(self, text="Add")
        self.addNodeButton.grid(row=1, column=2, pady=(2,5), padx=2)
        self.addNodeButton.configure(command=self.addNodeToList)
        # Remove node widgets:
        self.removeNodeLabel = ctk.CTkLabel(self, text="Remove:")
        self.removeNodeLabel.grid(row=2, column=0, pady=(2,5), padx=2)
        # Remove node entry using the CTkScrollableDropdown to autocomplete the node types
        self.removeNodeEntry = ctk.CTkEntry(self)
        self.removeNodeEntry.grid(row=2, column=1, pady=(0,5), padx=2)
        CTkScrollableDropdown(attach=self.removeNodeEntry, values=self.winfo_toplevel().logic_instance.all_nodes, command=lambda e: self.insert_method(e, self.removeNodeEntry), autocomplete=True) 
        self.removeNodeButton = ctk.CTkButton(self, text="Remove")
        self.removeNodeButton.grid(row=2, column=2, pady=(0,5), padx=2)
        self.removeNodeButton.configure(command=self.removeNodeToList)
        # List of included nodes widgets:
        self.listNodesLabel = ctk.CTkLabel(self, text="List of included nodes:")
        self.listNodesLabel.grid(row=3, column=0, columnspan=4, pady=5, padx=2)
        self.listNodesScrollFrame = ListNodesScrollFrame(self, is_include=True, possible_nodes=self.winfo_toplevel().logic_instance.all_nodes)
        self.listNodesScrollFrame.grid(row=4, column=0, columnspan=3, pady=(5, 2), padx=2, sticky="nsew")
        self.listNodesScrollFrame.columnconfigure(0, weight=1) 

    def addNodeToList(self):
        self.listNodesScrollFrame.addNode(self.addNodeEntry.get())
        self.addNodeEntry.delete(0, 'end')
    def removeNodeToList(self):
        self.listNodesScrollFrame.removeNode(self.removeNodeEntry.get())
        self.removeNodeEntry.delete(0, 'end')
        #TODO: Handle the case when the node is not in the list

class RemoveNodesFilterFrame(ctk.CTkFrame):
    """Frame that contains the nodes remove filter widgets, for the add there is a combobox with all the node types and a button to add the selected node. For the remove there is another combobox with all the selected nodes and a button to remove the selected node"""
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        # Configure the frame to handle 3 columns with equal weight and 5 rows with equal weight
        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.createWidgets()

    def insert_method(self, e, entry):
        entry.delete(0, 'end')
        entry.insert(0, e)

    def createWidgets(self):
        tmp_list_nodes = self.winfo_toplevel().logic_instance.all_nodes
        self.nodesLabel = ctk.CTkLabel(self, text="Excluded Node types:", font=("Arial", 14, "bold"))
        self.nodesLabel.grid(row=0, column=0, columnspan=3, pady=(10, 2), padx=2)
        # Add node widgets:
        self.addNodeLabel = ctk.CTkLabel(self, text="Add:")
        self.addNodeLabel.grid(row=1, column=0, pady=(2,5), padx=2)
        # Add node entry using the CTkScrollableDropdown to autocomplete the node types 
        self.addNodeEntry = ctk.CTkEntry(self)
        self.addNodeEntry.grid(row=1, column=1, pady=(0,5), padx=2)
        CTkScrollableDropdown(attach=self.addNodeEntry, values=tmp_list_nodes, command=lambda e: self.insert_method(e, self.addNodeEntry), autocomplete=True) 
        self.addNodeButton = ctk.CTkButton(self, text="Add")
        self.addNodeButton.grid(row=1, column=2, pady=(2,5), padx=2)
        self.addNodeButton.configure(command=self.addNodeToList)
        # Remove node widgets:
        self.removeNodeLabel = ctk.CTkLabel(self, text="Remove:")
        self.removeNodeLabel.grid(row=2, column=0, pady=(2,5), padx=2)
        # Remove node entry using the CTkScrollableDropdown to autocomplete the node types
        self.removeNodeEntry = ctk.CTkEntry(self)
        self.removeNodeEntry.grid(row=2, column=1, pady=(0,5), padx=2)
        CTkScrollableDropdown(attach=self.removeNodeEntry, values=self.winfo_toplevel().logic_instance.all_nodes, command=lambda e: self.insert_method(e, self.removeNodeEntry), autocomplete=True) 
        self.removeNodeButton = ctk.CTkButton(self, text="Remove")
        self.removeNodeButton.grid(row=2, column=2, pady=(0,5), padx=2)
        self.removeNodeButton.configure(command=self.removeNodeToList)
        # List of included nodes widgets:
        self.listNodesLabel = ctk.CTkLabel(self, text="List of excluded nodes:")
        self.listNodesLabel.grid(row=3, column=0, columnspan=4, pady=5, padx=2)
        self.listNodesScrollFrame = ListNodesScrollFrame(self, is_include=False, possible_nodes=tmp_list_nodes)
        self.listNodesScrollFrame.grid(row=4, column=0, columnspan=3, pady=(5, 2), padx=2, sticky="nsew")
        self.listNodesScrollFrame.columnconfigure(0, weight=1) 

    def addNodeToList(self):
        self.listNodesScrollFrame.addNode(self.addNodeEntry.get())
        self.addNodeEntry.delete(0, 'end')
    def removeNodeToList(self):
        self.listNodesScrollFrame.removeNode(self.removeNodeEntry.get())
        self.removeNodeEntry.delete(0, 'end')
        #TODO: Handle the case when the node is not in the list

class FilterScrollFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkScrollableFrame.__init__(self, master, *args, **kwargs, width=350, orientation="vertical")
        self.createWidgets()

    def createWidgets(self):
        self.architectureFilterFrame = ArchitectureFilterFrame(self)
        self.architectureFilterFrame.pack(side="top", fill="x")
        self.paramsFilterFrame = ParamsFilterFrame(self)
        self.paramsFilterFrame.pack(side="top", fill="x")
        self.nodesFilterFrame = AddNodesFilterFrame(self)
        self.nodesFilterFrame.pack(side="top", fill="x")
        self.removeNodesFilterFrame = RemoveNodesFilterFrame(self)
        self.removeNodesFilterFrame.pack(side="top", fill="x")
        self.originFilterFrame = OriginFilterFrame(self)
        self.originFilterFrame.pack(side="top", fill="x")

class OriginFilterFrame(ctk.CTkFrame):
    '''Frame containing the widgets to filter the networks by their origin benchmark (e.g. select only benchmarks from cifar100, acasxu, etc)'''
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.originNames = []
        self.originVars = []
        self.createWidgets()

    def interaction(self, id):
        if(id == -1):
            for i in range(len(self.originVars)):
                self.originVars[i].set(not self.originVars[id].get())
                #self.interaction(i)
            self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()
            return
        print(f"Origin {self.originNames[id]} selected -> {self.originVars[id].get()}")
        try:
            if(self.originVars[id].get()):
                self.winfo_toplevel().logic_instance.incuded_benchmarks.append(self.originNames[id])
            else:
                self.winfo_toplevel().logic_instance.incuded_benchmarks.remove(self.originNames[id])
                print(f"Removed {self.originNames[id]}")
        except:
            print("Error in the origin filter interaction :/")

    def createCell(self, row:int, column:int, name:str):
        self.originVars.append(ctk.BooleanVar(value=True))
        self.originVars[-1].trace_add('write', lambda *args: self.interaction(row + column))
        self.originNames.append(name)
        cellbox = ctk.CTkCheckBox(self.originFrame, text=name, variable=self.originVars[-1], font=("Arial", 10), command=lambda: self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview())
        cellbox.grid(row=row, column=column, pady=2, padx=2, sticky='ew')
    
    def createWidgets(self):
        self.originLabel = ctk.CTkLabel(self, text="Origin:", font=("Arial", 14, "bold"))
        self.originLabel.grid(row=0, column=0, pady=2, padx=2)
        all_origins = sorted(self.winfo_toplevel().logic_instance.possible_origins)

        all_var = ctk.BooleanVar(value=True)
        all_checkbox = ctk.CTkCheckBox(self, text="All", font=("Arial", 10), variable=all_var, command=lambda: self.interaction(-1))
        all_checkbox.grid(row=1, column=0, pady=2, padx=2, sticky='ew')

        self.originFrame = ctk.CTkFrame(self)
        self.originFrame.grid(row=2, column=0, pady=2, padx=2)
        self.originFrame.columnconfigure(0, weight=1)
        self.originFrame.columnconfigure(1, weight=1)

        for i, origin in enumerate(all_origins):
            self.createCell(i-i%2, i%2, origin)