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

        teach_button = Button(self.teach_page, text='Teach', command=self.teach_button_clicked)
        teach_button.grid(column=2, row=7)

        label = Label(self.teach_page, text='Training image', anchor='w')
        label.grid(column=0, row=0, columnspan=2, sticky='ew')
        self.entry_training_image = Entry(self.teach_page)
        self.entry_training_image.grid(column=0, row=1, sticky='ew')
        self.entry_training_image.focus_set()
        training_image_button = Button(self.teach_page, text='Open', command=lambda: Application.open_button_clicked(self.entry_training_image))
        training_image_button.grid(column=1, row=1)

        label = Label(self.teach_page, text='Number of repetitions', anchor='w')
        label.grid(column=0, row=2, columnspan=2, sticky='ew')
        self.entry_repetitions = Entry(self.teach_page)
        self.entry_repetitions.insert(0, '10000')
        self.entry_repetitions.grid(column=0, row=3, sticky='ew')

        label = Label(self.teach_page, text='Learning rate', anchor='w')
        label.grid(column=0, row=4, columnspan=2, sticky='ew')
        self.entry_rate = Entry(self.teach_page)
        self.entry_rate.insert(0, '0.5')
        self.entry_rate.grid(column=0, row=5, sticky='ew')

    def _init_compress_page(self):
        compress_button = Button(self.compress_page, text='Compress', command=self.compress_button_clicked)
        compress_button.grid(column=5, row=7)

        label = Label(self.compress_page, text='Image', anchor='w')
        label.grid(column=3, row=0, columnspan=2, sticky='ew')
        self.entry_image = Entry(self.compress_page)
        self.entry_image.grid(column=3, row=1, sticky='EW')
        self.entry_image.focus_set()
        image_button = Button(self.compress_page, text='Open', command=lambda: Application.open_button_clicked(self.entry_image))
        image_button.grid(column=4, row=1)

        label = Label(self.compress_page, text='Neural network', anchor='w')
        label.grid(column=3, row=2, columnspan=2, sticky='ew')
        self.entry_network = Entry(self.compress_page)
        self.entry_network.grid(column=3, row=3, sticky='EW')
        network_button = Button(self.compress_page, text='Open', command=lambda: Application.open_button_clicked(self.entry_network))
        network_button.grid(column=4, row=3)

        label = Label(self.compress_page, text='Bits', anchor='w')
        label.grid(column=3, row=4, columnspan=2, sticky='ew')
        self.entry_bits = Entry(self.compress_page)
        self.entry_bits.insert(0, '4')
        self.entry_bits.grid(column=3, row=5, sticky='EW')

    def _init_decompress_page(self):
        compress_button = Button(self.decompress_page, text='Decompress', command=self.decompress_button_clicked)
        compress_button.grid(column=8, row=7)

        label = Label(self.decompress_page, text='Compressed image', anchor='w')
        label.grid(column=6, row=0, columnspan=2, sticky='ew')
        self.entry_compressed_image = Entry(self.decompress_page)
        self.entry_compressed_image.grid(column=6, row=1, sticky='EW')
        self.entry_compressed_image.focus_set()
        compressed_image_button = Button(self.decompress_page, text='Open', command=lambda: Application.open_button_clicked(self.entry_compressed_image))
        compressed_image_button.grid(column=7, row=1)

        label = Label(self.decompress_page, text='Neural network', anchor='w')
        label.grid(column=6, row=2, columnspan=2, sticky='ew')
        self.entry_network2 = Entry(self.decompress_page)
        self.entry_network2.grid(column=6, row=3, sticky='EW')
        network_button2 = Button(self.decompress_page, text='Open', command=lambda: Application.open_button_clicked(self.entry_network2))
        network_button2.grid(column=7, row=3)

    def teach_button_clicked(self):
        try:
            output = tkFileDialog.asksaveasfilename(defaultextension='.mkm', filetypes=[('Neural network', '.mkm')])
            if output != '':
                compression.teach(output, self.entry_training_image.get(), int(self.entry_repetitions.get()), float(self.entry_rate.get()))
        except ValueError:
            tkMessageBox.showerror(message='Improper input values')

    def compress_button_clicked(self):
        try:
            output = tkFileDialog.asksaveasfilename(defaultextension='.zdp', filetypes=[('Compressed image', '.zdp')])
            if output != '':
                compression.compress(self.entry_image.get(), self.entry_network.get(), output, int(self.entry_bits.get()))
        except ValueError:
            tkMessageBox.showerror(message='Improper input values')
        except neural_network.NeuralNetworkException as exc:
            tkMessageBox.showerror(message='Cannot load neural network: ' + exc.message)
        except IOError as exc:
            tkMessageBox.showerror(message='Cannot load neural network: ' + exc.strerror)

    def decompress_button_clicked(self):
        try:
            output = tkFileDialog.asksaveasfilename(defaultextension='.bmp', filetypes=[('Bitmap', '.bmp')])
            if output != '':
                compression.decompress(self.entry_compressed_image.get(), self.entry_network2.get(), output)
        except ValueError:
            tkMessageBox.showerror(message='Improper input values')
        except neural_network.NeuralNetworkException as exc:
            tkMessageBox.showerror(message='Cannot load neural network: ' + exc.message)
        except IOError as exc:
            tkMessageBox.showerror(message='Cannot load neural network: ' + exc.strerror)

    @staticmethod
    def open_button_clicked(entry):
        filename = tkFileDialog.askopenfilename(filetypes=[('Bitmap', '.bmp'), ('Neural network', '.mkm'), ('Compressed image', '.zdp')])
        entry.delete(0, END)
        entry.insert(0, filename)