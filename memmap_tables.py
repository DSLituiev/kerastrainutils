import tables

def read_array(path):
    h5f = tables.open_file(path, 'r')
    no = h5f.list_nodes('/')[0]
    return no

