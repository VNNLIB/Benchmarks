import customtkinter as ctk
from CTkXYFrame import CTkXYFrame 
from CTkTable import CTkTable
from filter_gui import FilterScrollFrame

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

class MainFrame(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        ctk.CTkFrame.__init__(self, master, *args, **kwargs)
        self.createWidgets()

    def createWidgets(self):
        self.filterScrollFrame = FilterScrollFrame(self)
        self.filterScrollFrame.pack(side="left", fill="both")
        self.benchmarkScrollFrame = BenchmarkScrollFrame(self)
        self.benchmarkScrollFrame.pack(side="right", expand=True, fill="both")