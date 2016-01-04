from Tkinter import *
import tkMessageBox
import tkFileDialog
import ttk

import compression
import neural_network


class Application(Tk):
    def __init__(self):
        Tk.__init__(self, None)
        self.title('zdp')
        self.resizable(False, False)
        nb = ttk.Notebook(self)

        self.teach_page = ttk.Frame(nb)
        self.teach_page.grid()
        self.compress_page = ttk.Frame(nb)
        self.compress_page.grid()
        self.decompress_page = ttk.Frame(nb)
        self.decompress_page.grid()

        nb.add(self.teach_page, text='Teach')
        nb.add(self.compress_page, text='Compress')
        nb.add(self.decompress_page, text='Decompress')
        nb.pack(expand=1, fill='both')

        self._init_teach_page()
        self._init_compress_page()
        self._init_decompress_page()

    def _init_teach_page(self):
        """Initialize entries and buttons in 'Teach' tab.
           Assign actions to buttons and set default values.
        """
        teach_button = Button(self.teach_page, text='Run',
                              command=lambda: Application.run_button_clicked('.mkm', [('Neural network', '.mkm')], self.teach_page, self.do_teach))
        teach_button.grid(column=2, row=9)

        label = Label(self.teach_page, text='Training image', anchor='w')
        label.grid(column=0, row=0, columnspan=2, sticky='ew')
        self.training_image_entry = Entry(self.teach_page)
        self.training_image_entry.grid(column=0, row=1, sticky='ew')
        self.training_image_entry.focus_set()
        training_image_button = Button(self.teach_page, text='Open',
                                       command=lambda: Application.open_button_clicked(self.training_image_entry, [('Bitmap', '.bmp')]))
        training_image_button.grid(column=1, row=1)

        label = Label(self.teach_page, text='Number of repetitions', anchor='w')
        label.grid(column=0, row=2, columnspan=2, sticky='ew')
        self.repetitions_entry = Entry(self.teach_page)
        self.repetitions_entry.insert(0, '30000')
        self.repetitions_entry.grid(column=0, row=3, sticky='ew')

        label = Label(self.teach_page, text='Learning rate', anchor='w')
        label.grid(column=0, row=4, columnspan=2, sticky='ew')
        self.rate_entry = Entry(self.teach_page)
        self.rate_entry.insert(0, '0.5')
        self.rate_entry.grid(column=0, row=5, sticky='ew')

        label = Label(self.teach_page, text='Hidden layer size', anchor='w')
        label.grid(column=0, row=6, columnspan=2, sticky='ew')
        self.layer_size_entry = Entry(self.teach_page)
        self.layer_size_entry.insert(0, '32')
        self.layer_size_entry.grid(column=0, row=8, sticky='ew')

    def _init_compress_page(self):
        """Initialize entries and buttons in 'Compress' tab.
           Assign actions to buttons and set default values.
        """
        compress_button = Button(self.compress_page, text='Run',
                                 command=lambda: Application.run_button_clicked('.zdp', [('Compressed image', '.zdp')], self.compress_page,
                                                                                self.do_compress))
        compress_button.grid(column=2, row=9, sticky='sw')

        label = Label(self.compress_page, text='Image', anchor='w')
        label.grid(column=0, row=0, columnspan=2, sticky='ew')
        self.image_entry = Entry(self.compress_page)
        self.image_entry.grid(column=0, row=1, sticky='EW')
        self.image_entry.focus_set()
        image_button = Button(self.compress_page, text='Open',
                              command=lambda: Application.open_button_clicked(self.image_entry, [('Bitmap', '.bmp')]))
        image_button.grid(column=1, row=1)

        label = Label(self.compress_page, text='Neural network', anchor='w')
        label.grid(column=0, row=2, columnspan=2, sticky='ew')
        self.network_entry = Entry(self.compress_page)
        self.network_entry.grid(column=0, row=3, sticky='EW')
        network_button = Button(self.compress_page, text='Open',
                                command=lambda: Application.open_button_clicked(self.network_entry, [('Neural network', '.mkm')]))
        network_button.grid(column=1, row=3)

        label = Label(self.compress_page, text='Bits', anchor='w')
        label.grid(column=0, row=4, columnspan=2, sticky='ew')
        self.bits_entry = Entry(self.compress_page)
        self.bits_entry.insert(0, '4')
        self.bits_entry.grid(column=0, row=5, sticky='EW')

    def _init_decompress_page(self):
        """Initialize entries and buttons in 'Decompress' tab.
           Assign actions to buttons and set default values.
        """
        compress_button = Button(self.decompress_page, text='Run',
                                 command=lambda: Application.run_button_clicked('.bmp', [('Bitmap', '.bmp')], self.decompress_page,
                                                                                self.do_decompress))
        compress_button.grid(column=2, row=9)

        label = Label(self.decompress_page, text='Compressed image', anchor='w')
        label.grid(column=0, row=0, columnspan=2, sticky='ew')
        self.compressed_image_entry = Entry(self.decompress_page)
        self.compressed_image_entry.grid(column=0, row=1, sticky='EW')
        self.compressed_image_entry.focus_set()
        compressed_image_button = Button(self.decompress_page, text='Open',
                                         command=lambda: Application.open_button_clicked(self.compressed_image_entry, [('Compressed image', '.zdp')]))
        compressed_image_button.grid(column=1, row=1)

        label = Label(self.decompress_page, text='Neural network', anchor='w')
        label.grid(column=0, row=2, columnspan=2, sticky='ew')
        self.network_entry2 = Entry(self.decompress_page)
        self.network_entry2.grid(column=0, row=3, sticky='EW')
        network_button2 = Button(self.decompress_page, text='Open',
                                 command=lambda: Application.open_button_clicked(self.network_entry2, [('Neural network', '.mkm')]))
        network_button2.grid(column=1, row=3)

    @staticmethod
    def run_button_clicked(defaultextension, filetypes, parent, action):
        """Open dialog window so that user can choose output file.
           Call action and show error message if any was occurred.
        """
        output = tkFileDialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)
        if output != '':
            label = Label(parent, text='In progress...', anchor='w')
            label.grid(column=0, row=9, columnspan=2, sticky='ew')
            label.update()
            try:
                action(output)
            except ValueError:
                tkMessageBox.showerror(message='Improper input values')
            except neural_network.NeuralNetworkException as exc:
                tkMessageBox.showerror(message='Cannot load neural network: ' + exc.message)
            except compression.ZdpException as exc:
                tkMessageBox.showerror(message=exc.message)
            except IOError as exc:
                tkMessageBox.showerror(message='Cannot load neural network: ' + exc.strerror)
            finally:
                label.destroy()

    @staticmethod
    def open_button_clicked(entry, filetypes=[]):
        """Open dialog window and place chosen file path in entry"""
        filename = tkFileDialog.askopenfilename(filetypes=filetypes)
        entry.delete(0, END)
        entry.insert(0, filename)

    def do_teach(self, output):
        compression.teach(output, self.training_image_entry.get(), int(self.repetitions_entry.get()), float(self.rate_entry.get()),
                          int(self.layer_size_entry.get()))

    def do_compress(self, output):
        compression.compress(self.image_entry.get(), self.network_entry.get(), output, int(self.bits_entry.get()))

    def do_decompress(self, output):
        compression.decompress(self.compressed_image_entry.get(), self.network_entry2.get(), output)