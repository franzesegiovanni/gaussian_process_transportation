
def save_array_as_latex(array, filename):
    with open(filename, 'w') as f:
        rows = len(array)
        cols = len(array[0])

        # Write the table header
        f.write("\\begin{tabularx}{\\linewidth}{|c|" + "X|" * cols + "}\n")
        f.write("\\hline\n")

        # Write the column headers
        f.write(" & ")
        f.write(" & ".join(["Column " + str(i + 1) for i in range(cols)]))
        f.write(" \\\\\n")
        f.write("\\hline\n")

        # Write the table content
        for i in range(rows):
            f.write("Row " + str(i + 1) + " & ")
            f.write(" & ".join("{:.2f}".format(x) for x in array[i]))
            f.write(" \\\\\n")
            f.write("\\hline\n")

        # Write the table footer
        f.write("\\end{tabularx}")
