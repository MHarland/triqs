import re, os

def decay(s) :
    for tok in ['const ', 'const&', '&&', '&'] :
        s = re.sub(tok,'',s)
    return s.strip()

def process_param_type(t):
    if t.name in class_list: # has a link
       d = decay(t.name)
       return t.name.replace(d,":ref:`%s <%s>`"%(d,d))
    else: 
       return t.name

def process_rtype(t) :
    tname =  re.sub(r"\s*typename\s+std\d*::enable_if<(.*),(.*)>::type", r"requires(\1)\n \2 ", t.name)
    return tname

def make_synopsis_template_decl(tparams) : 
    if not tparams: return ''
    targs = ', '.join("%s %s"%(pp[0],pp[1]) + (" = %s"%pp[2] if (len(pp)==3 and pp[2]) else '') for pp in tparams)
    return "template<%s>"%targs

def add_linebreaks(s, num_char=80):
    """ add linebreaks every num_char characters in string s (if possible i.e if whitespace)"""
    char_count=0
    final_s=''
    for w in s.split(' '):
     char_count += len(w)
     if char_count < num_char: final_s += w+' '
     else: final_s += '\n   '+w+' '; char_count=len(w)
    return final_s

def make_synopsis(m, decal):
    #assert not m.tparams, "template functions "
    syn = m.doc_elements['synopsis']
    if syn : return [syn]
    s = " {name} ({args}) {const}; "
    if not m.is_constructor :
      s = process_rtype(m.rtype) + s
    s = make_synopsis_template_decl(m.tparams) + "\n" + s
    args = ', '.join( ["%s %s"%(process_param_type(t),n) + (" = %s"%d if d else "") for t,n,d in m.params])
    s = s.format(args = args, name = m.name.strip(), const = m.const)
    r = [x.strip() for x in s.split('\n')]
    L= [x for x in r if x]
    L_lb = [add_linebreaks(x) for x in L]
    return L_lb

def make_synopsis_list(m_list):
    if not m_list: return ''
    decal = 4
    signs = [ make_synopsis(x, decal) for x in m_list]
    m = max( max (len(l) for l in s) for s in signs)
    tab = ("\n" + decal*' ')
    form =  '{:<%s}    {:<%s}'%(m, 3)
    lines =  []
    for n,s in enumerate(signs) :
       for p,l in enumerate(s):
          if p==0 :
            lines += [ '',  form.format(l, "(%s)"%(n+1) if len(m_list)>1 else '' )]
          else :
            lines.append(form.format(l,''))
    return  decal*' ' + tab.join(lines[1:])

def make_table(head_list, list_of_list):
    l = len (head_list)
    lcols = [len(x) for x in head_list]
    for li in list_of_list : # compute the max length of the columns
        lcols = [ max(len(x), y) for x,y in zip(li, lcols)]
    form =  '| ' + " | ".join("{:<%s}"%x for x in lcols).strip() + ' |'
    header= form.format(*head_list)
    w = len(header)
    sep = '+' + '+'.join((x+2) *'-' for x in lcols) + '+'
    sep1 = sep.replace('-','=')
    r = [sep, header, sep1]
    for li in list_of_list: r += [form.format(*li), sep] 
    return '\n'.join(r)

def prepare_example(filename, decal):
    """From the filename, prepare the doc1, doc2, before and after the code
       and compute the lineno of the code for inclusion"""
    filename += ".cpp"
    if not os.path.exists(filename) : 
        #print "example file %s (in %s) does not exist"%(filename,os.getcwd())
        return None, None, None, 0, 0 
    ls = open(filename).read().strip().split('\n')
    r = [i for i, l in enumerate(ls) if not (re.match(r"^\s*/?\*",l) or re.match("^\s*//",l))]
    s, e = r[0],r[-1]+1
    assert r == range(s,e)
    def cls(w) : 
        w = re.sub(r"^\s*/?\*\s?/?",'',w)
        w = re.sub(r"^\s*//\s?",'',w)
        return w
    doc1 = '\n'.join(cls(x) for x in ls[0:s])
    code = '\n'.join(decal*' ' + x.strip() for x in ls[s:e])
    doc2 = '\n'.join(cls(x) for x in ls[e:])
    return code, doc1, doc2, s, e
