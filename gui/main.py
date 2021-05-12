import tkinter as tk
from tkinter import LEFT, TOP, ALL

from constants import BG_COLOR, BG_TEXTBOX, BG_BUTTON, BG_BUTTON_ACTIVE, TILETYPE_ICON_SIZE, BOX_COLOR, AGENT_COLOR, OBSTACLE_COLOR, FREE_COLOR, BOX, AGENT, OBSTACLE, FREE

def main():
	w = Window()
	w.init_user_interface()

class Popup_Window:
	root = None
	
	def __init__(self, option):
		self.option = option
		self.value = 0
		self.top= tk.Toplevel()
		self.top.grab_set()
		self.top.attributes('-topmost', 'true')
		self.top.title("Input")
		self.top.geometry('+%d+%d' % (1000, 0))
		if option == "Weight":	
			self.label = tk.Label(self.top,text="Weight")

		else :
			self.label = tk.Label(self.top,text="Capacity")
		self.label.pack(side = TOP)
		self.entry = tk.Entry(self.top)
		self.entry.pack(side = TOP)
		self.entry.focus()
		self.button = tk.Button(self.top,text='Ok',command=self.cleanup)
		self.button.pack(side = TOP)
		self.root.wait_window(self.top)

	def cleanup(self):
		self.value = self.entry.get()
		try:
			self.value = int(self.value)
		except:
			message(self.option +' must be an integer!', wait_time=2000)
			return 

		message(self.option +' Recorded!', wait_time=1000)
		self.top.destroy()
		
class Window():
	def __init__(self):
		self.root = tk.Tk()
		self.root.geometry("1366x768")
		self.root.configure(background=BG_COLOR, cursor="arrow")

		Popup_Window.root = self.root
		Create_Canvas.root = self.root

	def init_user_interface(self):
		self.left_frame = tk.Frame(self.root)
		self.left_frame.configure(background=BG_COLOR)
		self.left_frame.pack(side = LEFT)

		self.right_frame = tk.Frame(self.root)
		self.right_frame.configure(background=BG_COLOR)
		self.right_frame.pack(side = LEFT)

		self.left_up_frame = tk.Frame(self.left_frame)
		self.left_up_frame.configure(background=BG_COLOR)
		self.left_up_frame.pack(side = TOP)

		self.left_down_frame = tk.Frame(self.left_frame)
		self.left_down_frame.configure(background=BG_COLOR)
		self.left_down_frame.pack(side = TOP)

		# add logo to top left frame
		logo = tk.PhotoImage(file='robot.gif')
		tk.Label(self.left_up_frame, image=logo, bd=0).pack()

		# create row and col entry labels and textboxes
		inputs = [
			{
				'id': 'rows',
				'text':'Rows',
			},
			{
				'id': 'cols',
				'text':'Cols',
			},
		]
		self.entries = {}
		for index, entry in enumerate(inputs):
			tk.Label(self.left_down_frame, text=entry['text'], borderwidth=1, height=5, width=10, bg=BG_COLOR).grid(row=index,column=0)
			curr_entry = tk.Entry(self.left_down_frame, bg=BG_TEXTBOX, width=10)
			curr_entry.grid(row=index, column=1)
			self.entries[entry['id']] = curr_entry
		self.entries['rows'].focus()

		# create tiletype labels and icons
		self.tiletypes = ['Box', 'Agent', 'Obstacle', 'Free']
		for index, tiletype in enumerate(self.tiletypes):
			tk.Label(self.left_down_frame, text=tiletype, borderwidth=1, height=5, width=10, bg=BG_COLOR).grid(row=index + 2,column=0)
			Create_Canvas(self.left_down_frame, index, index + 2, 1, TILETYPE_ICON_SIZE, bg=FREE_COLOR)

		# create buttons
		self.buttons = [
			{
				'text':'Save !',
				'callback':save_to_file,
			},
			{
				'text':'Run !',
				'callback':run_simulation,
			},
			{
				'text':'Grid !',
				'callback':lambda: draw_grid(self.entries['rows'], self.entries['cols'], self.right_frame),
			},
		]
		for index, button in enumerate(self.buttons):
			tk.Button(self.left_down_frame, bg=BG_BUTTON, activebackground=BG_BUTTON_ACTIVE , text=button['text'], command=button['callback'], height=3, width=14).grid(row=6,column=index)

		self.root.mainloop()

class Create_Canvas:
	grid = None
	waiting_for_goal = False
	agent_list = []
	box_list = []
	root = None

	def __init__(self, frame, shape_id, row, col, size, bg=BG_COLOR):
		self.frame = frame
		self.row = row
		self.col = col
		self.shape_id = shape_id
		self.size = size
		self.agent_no = 0
		self.agent_label = ""
		self.canvas = tk.Canvas(frame, bg=bg, height=self.size, width=self.size, bd = 0)

		self.shape_draw_functions = [
			{
				'function':self.canvas.create_rectangle,
				'color':BOX_COLOR,
			},
			{
				'function':self.canvas.create_oval,
				'color':AGENT_COLOR,
			},
			{
				'function':self.canvas.create_rectangle,
				'color':OBSTACLE_COLOR,
			},
			{
				'function':self.canvas.create_rectangle,
				'color':FREE_COLOR,
			},
		]
		draw_function = self.shape_draw_functions[self.shape_id]
		self.shape = draw_function['function'](0, 0, self.size, self.size, fill=draw_function['color'], outline="")
		self.canvas.grid(row=self.row, column=self.col)
		self.bind()
		
	def bind(self):
		events = {
			'<Button1-Motion>':self.drag,
			'<ButtonRelease-1>':self.drop,
			'<Enter>':self.enter,
			'<Leave>':self.leave,
		}
		for event, callback in events.items():
			self.canvas.bind(event, callback)
		if self.size == 30:
			self.canvas.bind('<Button-1>', self.get_goal)
		self.move_flag = False

	def drag(self, event):
		# check or press or drag
		# if press, create tmp canvas else, move tmp canvas
		new_xpos, new_ypos = event.x, event.y

		Create_Canvas.root.configure(cursor=('@/usr/include/X11/bitmaps/star', '/usr/include/X11/bitmaps/star', 'black', 'white'))
		if self.move_flag:
			new_xpos, new_ypos = event.x, event.y
			self.mouse_xpos = new_xpos
			self.mouse_ypos = new_ypos
			# drag code will be added here
		else:
			self.move_flag = True
			self.mouse_xpos = event.x
			self.mouse_ypos = event.y

	def drop(self, event):
		self.move_flag = False
		Create_Canvas.root.configure(cursor="arrow")
		widget = self.canvas.winfo_containing(event.x_root, event.y_root)
		row = widget.grid_info()['row']
		col = widget.grid_info()['column']		
		gm = Create_Canvas.grid.grid_map
		cell = gm[row][col]
		cell_canvas = cell.canvas
		size = cell.size
		
		if self.shape_id == BOX:
			Create_Canvas.waiting_for_goal = True
			if cell.shape_id == BOX:
				message('Cannot place multiple boxes on same location!', wait_time=3000)
				return
			cell.shape_id = self.shape_id
			cell.canvas.delete(ALL)
			draw_function = cell.shape_draw_functions[BOX]
			cell.shape = draw_function['function'](0, 0, size, size, fill=draw_function['color'], outline="")
			if cell.agent_no != 0:
				cell.agent_label = cell.canvas.create_text(size / 2, size / 2, text=str(cell.agent_no))		
			Create_Canvas.box_list.append([str(row), str(col), "-1", "-1", self.popup("Weight")])

		elif self.shape_id == AGENT:
			cell.canvas.delete(ALL)
			cell.agent_no = cell.agent_no + 1
			if cell.shape_id == BOX:
				draw_function = cell.shape_draw_functions[BOX]
				cell.shape = draw_function['function'](0, 0, size, size, fill=draw_function['color'], outline="")
			else:
				cell.shape_id = self.shape_id 
				draw_function = cell.shape_draw_functions[AGENT]
				cell.shape = draw_function['function'](0, 0, size, size, fill=draw_function['color'], outline="")
			if cell.agent_no != 0:
				cell.agent_label = cell.canvas.create_text(size / 2, size / 2, text=str(cell.agent_no))						
			Create_Canvas.agent_list.append([str(row), str(col), self.popup("Capacity")])
		elif self.shape_id == OBSTACLE:
			print(row, col)
			print('boxes')
			print(*Create_Canvas.box_list, sep='\n')
			on_box_goal = any(tuple(box[2:4]) == (str(row), str(col)) for box in Create_Canvas.box_list)
			if on_box_goal:
				message("Cannot have obstacle on box goal!", wait_time=3000)
			else:
				cell.canvas.delete(ALL)
				cell.shape_id = self.shape_id
				draw_function = cell.shape_draw_functions[OBSTACLE]
				cell.shape = draw_function['function'](0, 0, size, size, fill=draw_function['color'], outline="")
				cell.agent_no = 0

		elif self.shape_id == FREE:
			cell.canvas.delete(ALL)
			cell.shape_id = self.shape_id
			draw_function = cell.shape_draw_functions[FREE]
			cell.shape = draw_function['function'](0, 0, size, size, fill=draw_function['color'], outline="")
			cell.agent_no = 0

		self.bind()
	
	def popup(self, option):
		p = Popup_Window(option)
		while p.value <= 0:
			message("Positive Integer weight must be entered!", wait_time=2000)
			p = Popup_Window(option)
		return p.value
		
	def enter(self, event):
		pass
	
	def leave(self, event):
		pass

	def get_goal(self, event):
		if Create_Canvas.waiting_for_goal:
			on_obstacle = Create_Canvas.grid.grid_map[self.row][self.col].shape_id == OBSTACLE
			
			if not on_obstacle:
				Create_Canvas.waiting_for_goal = False
				message("Goal Recorded!")
				Create_Canvas.box_list[-1][2] = str(self.row)
				Create_Canvas.box_list[-1][3] = str(self.col)
			else:
				message("Cannot have goal on obstacle!", wait_time=2000)

def message(msg, wait_time=1000):
		top = tk.Toplevel()
		top.title('Message')
		tk.Message(top, text=msg, padx=50, pady=50).pack()
		top.after(wait_time, top.destroy)	

class Grid:
	def __init__(self, r, c):
		self.rows = r
		self.cols = c
		self.grid_map = [[0 for x in range(c)] for y in range(r)] 

	@staticmethod
	def set_up_next_grid(self, tmp_canvas, rows, col):
		self.grid_map[rows][col] = tmp_canvas

	def print_map(self):
		for r in range(self.rows):
			print("Row no", r)
			for c in range(self.cols):
				print(self.grid_map[r][c].shape_id , self.grid_map[r][c].agent_no)

def draw_grid(rows_entry, cols_entry, right_frame):	
	r = rows_entry.get()	
	c = cols_entry.get()	
	rows = int(r)
	cols = int(c)
	if not (1 <= rows <= 20 and 1 <= cols <= 20):
		message('Max grid size is 20x20!', wait_time=2000)
		return
	
	Create_Canvas.grid = Grid(rows, cols)
	for r in range(rows):
		for c in range(cols):
			tmp_canvas = Create_Canvas(right_frame, 3, r, c, 30, bg=FREE_COLOR)
			Grid.set_up_next_grid(Create_Canvas.grid, tmp_canvas, r, c)
						
def save_to_file():
	if Create_Canvas.waiting_for_goal:
		message('Choose box goal before saving!', wait_time=2000)
		return

	with open("obstacle_map", 'w') as f:
		for r in range(Create_Canvas.grid.rows):
			for c in range(Create_Canvas.grid.cols):
				if Create_Canvas.grid.grid_map[r][c].shape_id == 2	:
					print('#', end='', file=f)
				else :			
					print('.', end='', file=f)
			print(file=f)

	with open("boxes_data", 'w') as f:
		for box in Create_Canvas.box_list:
			print(*box, sep='\t', file=f)

	with open('agents_data', 'w') as f:
		for agent in Create_Canvas.agent_list:
			print(*agent, sep='\t', file=f)				

def run_simulation():
	save_to_file()
	pass

if __name__ == '__main__':
	main()
