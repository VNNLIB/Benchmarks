import customtkinter as ctk
from logic import *

# Thanks to Akascape: https://github.com/Akascape/CTkXYFrame, https://github.com/Akascape/CTkScrollableDropdown, https://github.com/Akascape/CtkTable
from CTkXYFrame import * 
from CTkScrollableDropdown import *
from CTkTable import * 

# GraphicalClass
# graphicalObject
# graphicalMethod()
# logical_method()

class SupportFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        self.createWidgets()

    def calculate(self):
        print("Calculate button clicked")
        self.winfo_toplevel().mainFrame.benchmarkScrollFrame.updateTabview()
        self.winfo_toplevel().logic_instance.prova()
    
    def createWidgets(self):
        print("Creating widgets in SupportFrame")
        self.calculateButton = ctk.CTkButton(self, text="Calculate", command=self.calculate, width=0)
        self.calculateButton.grid(row=0, column=0, padx=4, pady=5)

        self.optionButton = ctk.CTkButton(self, text="Options", width=100, corner_radius=8)
        self.optionButton.grid(row=0, column=1, padx=4, pady=5)

        self.helpButton = ctk.CTkButton(self, text="Help", width=100, corner_radius=8)
        self.helpButton.grid(row=0, column=2, padx=4, pady=5)

class BenchmarkScrollFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs, border_color="#000000", border_width=2)
        self.columns_names = ['onnx', 'architecture', 'benchmark', 'n_params', 'node_types']
        self.table_dataset = []
        self.createWidgets()

    def createWidgets(self):#TODO: Handle the tabulation of the columns 
        self.scrollFrame = CTkXYFrame(self)
        self.scrollFrame.pack(expand=True, fill="both")
        self.table_tabview = None
        self.list_tabs = []
        self.list_tabs_names = []
        self.updateTabview()

    def loadClickedTab(self):
        print("CALLED")
        id_tab = int(self.table_tabview.get().replace("Tab ", "").strip())
        if self.list_cached_tabs[id_tab]:
            print(f"Tab {id_tab} already cached")
            return
        print(f"Loading tab {id_tab}")

        self.list_cached_tabs[id_tab] = True
        lower_bound = id_tab*20
        upper_bound = (id_tab+1)*20
        # Create the table with the data
        vals = [self.columns_names]
        for row in self.table_dataset[lower_bound:min(upper_bound, len(self.table_dataset))]:
            vals.append(row)
        table = CTkTable(self.list_tabs[id_tab], row=upper_bound-lower_bound+1, column=len(self.columns_names), fg_color=self._bg_color, bg_color=self._bg_color, values=vals)
        table.pack(expand=True, fill="both")

    def updateTabview(self):
        if self.table_tabview:
            self.table_tabview.destroy()
        #TODO: Refactor this part to avoid the creation of new tabs every time
        # TODO: resolve the problem that the tabview is not matching the parent frame when the window is resized (and the space is enough)
        self.table_tabview = ctk.CTkTabview(self.scrollFrame, command=self.loadClickedTab, fg_color=self._bg_color, bg_color=self._bg_color)
        self.table_tabview.pack(expand=True, fill="both")
        
        self.table_dataset = self.winfo_toplevel().logic_instance.get_benchmarks_sample()
        n_tabs = len(self.table_dataset)//20 + 1 if len(self.table_dataset)%20 != 0 else len(self.table_dataset)//20
        self.list_cached_tabs = [False for _ in range(n_tabs)]
        self.list_tabs = [self.table_tabview.add(name) for name in [f"Tab {i}" for i in range(n_tabs)]]
        if n_tabs > 0:
            self.loadClickedTab()#Load the first tab
        



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
        
        print(f"Changed to {self.isFullyConnectedIncluded.get()} {self.isConvolutionalIncluded.get()} {self.isResidualIncluded.get()}")
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
        if node == "all" or 'all' in self.nodeInList:
            if not self.is_include:
                self.nodeInList = ['all'] #Empty the list ('all' includes all the nodes itself)
                self.updateList()
                return
            else:
                print("Node 'all' cannot be added to the included nodes")

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
        self.createWidgets()

    def insert_method(self, e, entry):
        entry.delete(0, 'end')
        entry.insert(0, e)

    def createWidgets(self):
        self.nodesLabel = ctk.CTkLabel(self, text="Included Node types:", font=("Arial", 14, "bold"))
        self.nodesLabel.grid(row=0, column=0, columnspan=3, pady=(10, 2), padx=2)
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
        tmp_list_nodes = self.winfo_toplevel().logic_instance.all_nodes+["all"]
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

class MainFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs)
        self.createWidgets()

    def createWidgets(self):
        self.filterScrollFrame = FilterScrollFrame(self)
        self.filterScrollFrame.pack(side="left", fill="both")
        self.benchmarkScrollFrame = BenchmarkScrollFrame(self)
        self.benchmarkScrollFrame.pack(side="right", expand=True, fill="both")

class GUI(ctk.CTk):
    def __init__(self, logic_instance):
        ctk.CTk.__init__(self)
        self.title("Instances Generator")
        self.logic_instance = logic_instance
        self.geometry("800x400")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        self.minsize(350, 395)
        self.createWidgets()
    
    def createWidgets(self):
        self.supportFrame = SupportFrame(self)
        self.supportFrame.pack(side="top", fill="x")
        self.mainFrame = MainFrame(self)
        self.mainFrame.pack(side="top", expand=True, fill="both")

def main():
    logic_instance = logic()
    app = GUI(logic_instance)
    app.mainloop()
    

if __name__ == "__main__":
    main()