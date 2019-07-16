import os
import matplotlib.pyplot as plt

class utils:
    """
    This class consists for all the functions recursively required during model development.
    """
    
    # function to plot model training logs 
    def plot_training_history(self, history_dict, plot_val=True, accuracy=True, chart_type="--o"):
        """
        This used to plot training and validation stats post model training
        
        Arguments:
        history_dict {dict} -- keras post model training dictionary with comprehensive training stats
        plot_val {bool} -- boolean represent whether to include validation stats in graph
        accuracy {bool} -- boolean represent whether to include accuracy stats in graph
        chart_type {str} -- string represent the type of the graphs
        """
        
        if accuracy:
            acc = history_dict['acc']
        loss = history_dict['loss']
      
        if plot_val:
            if accuracy:
                val_acc = history_dict['val_acc']
            val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
    
        # visualize model training
        epochs = range(1, len(acc) + 1)
        fig, axs = plt.subplots(1, 2,figsize=(15,5))
        axs[0].plot(epochs, loss, chart_type, label='Training loss')
        if plot_val:
            axs[0].plot(epochs, val_loss, chart_type, label='Validation loss')
            axs[0].set_title('training & validation loss')
        else:
            axs[0].set_title('training loss')
        axs[0].legend()
        
        if accuracy:
            axs[1].plot(epochs, acc, chart_type, label='Training acc')
            if plot_val:
                axs[1].plot(epochs, val_acc, chart_type, label='Validation acc')
                axs[1].set_title('training & validation accuracy')
            else:
                axs[1].set_title('training accuracy')                      
            axs[1].legend()
        plt.show()
        plt.close()
        return
                                    
    # function to plot image without border    
    def plot_large_image_without_borders(self, path_to_image):
        """
        This function is used to display an image without border in an ipython notebook
        Argument:
        path_to_image {str}: path to image file
        """
        # change the figure size
        fig2 = plt.figure(figsize = (15,15)) 
        # create a 5 x 5 figure
        ax3 = fig2.add_subplot(111)
        ax3.imshow(cv2.imread(path_to_image), interpolation='none')
        plt.axis('off')
        plt.show()
    
    def count_files_in_directory(self, folder_path):
        total = 0
        for root, dirs, files in os.walk(folder_path):
            total += len(files)
        return total