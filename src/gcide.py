
import re
import string

import xml.etree.ElementTree as ET

global_pos_list = set()

split_words = ["amp;", "or", ",", "and"]

def reduce_pos(pos):

    pos = pos.strip()

    if pos.startswith("2d") or pos.startswith("3d"):
        return "o"
    if pos in ["v.", "a.", "adv.", "conj.", "prep.", "interj.", "pron."]:
        return pos[:-1]
    if pos in ["v. t.", "v. i.", 'p. pr.', 'p. p.', "imp.", "p. p", "p.", "pret.", 
               'infinitive.', 'participle', 'imp. sing.', 'obs. p. p.', 
               'present participle', 'imperative.', 'v. impers.', 'strong imp.',
               'archaic p. p.', 'prop. v. i.', 'imp. (', 'obs. imp. pl.',
               'v. impersonal', 'v. imperative.', 'pres. indic.', 'prop. v. t.',
               'pres. indic. 1st']:
        return "v"
    if "n." in pos:
        return "n"
    if pos in ["adj. prenom.", "prop. a.", 'prop. adj.', 'pred. adj.', "adj.", "p.a.",
               'p. a.', 'pred. a.', 'postnominal adj.', 'a.;', 'a. .', 'a. fem.',
               'pr. a.', 'a. superl.', 'a. compar.']:
        return "a"
    if pos in ["superl.", "compar.", "pref.", 'ads.', 'phrase', 'imperative sing.', 
               'prefix.', 'suffix.', 'def. art.', 'third pers. sing. pres.', 'phr.',
               'exclamation', 'suff.', 'suffix', 'definite article.',
               'obs. 3d pers. sing. pres.', 'comp.', 'abl.']:
        return "o"
    if pos in ['pl. indic. pr.']:
        return "pron"
    if pos in ["interrog. adv."]:
        return "adv"
    if pos in ["t.", "i.", "pl.", 'pl. in usage', '1st', 'fem.', 'auxiliary.', 'imp. pl.', 
               'obs. imp.', 'auxiliary', 'pres.', 'contrac.', 'pl. pres.',
               'syntactically sing.', "sing.", 'sing. pres.', 'but sing.'] + split_words:
        return None
    return pos


def get_pos_pair(p_elem):
    """Given a p-element, return a tuple with (entry, pos), or None"""
    
    entry_text = entry_pos = None
    
    for elt in p_elem:
        if elt.tag == 'ent':
            entry_text = elt.text.lower()
        elif elt.tag == 'pos':
            entry_pos = elt.text
        elif elt.tag == 'def' and not entry_pos:
            for elt_child in elt:
                if elt_child.tag == 'pos':
                    entry_pos = elt_child.text

    if entry_text and entry_pos:
        return (entry_text, entry_pos)
    else:
        # It's possible entry could have a value, but pos doesn't
        return (None, None)



def extract_pos(filename):

    pos_dict = {}

    with open(filename, 'r') as fhand:
        slurped = fhand.read()
    # Parser expects a single top-level node
    wrapped = "<rootnode>" + slurped + "</rootnode>"
    # Reduce all entities to nothing
    root = ET.fromstring(re.sub('&([A-Za-z0-9_]+;)', '\\1', wrapped))

    for p_elem in root:
        entry, pos = get_pos_pair(p_elem)
        if entry:
            pos_mini_list = set(filter(None, map(reduce_pos, re.split(' *(' + "|".join(split_words) + ') *', pos))))
            #pos_mini_list = [pos]
            for pos_item in pos_mini_list:
                global_pos_list.add(pos_item)
            if entry in pos_dict:
                pos_dict[entry].update(pos_mini_list)
            else:
                pos_dict[entry] = pos_mini_list
    
    return pos_dict


def get_full_pos_dict():
    master_pos_dict = {}
    for let in string.ascii_lowercase:
        fname = 'gcide_xml-0.51/xml_files/gcide_{}.xml'.format(let)
        print("Processing", fname)
        master_pos_dict.update(extract_pos(fname))
    return master_pos_dict


if __name__ == "__main__":

    master_pos_dict = get_full_pos_dict()

    for query in ["dog", "jealousy", "cast", "help", "she", "ran", "earth", "jump"]:
        try:
            print(query, master_pos_dict[query])
        except KeyError:
            print(query, "not in list")

    print(global_pos_list)