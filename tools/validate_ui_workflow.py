import json, sys
oi = json.load(open(sys.argv[1], encoding='utf-8'))
d  = json.load(open(sys.argv[2], encoding='utf-8'))
nodes = {n["id"]: n for n in d["nodes"]}
errs, warns = [], []
VIRTUAL = {"SetNode","GetNode","Note"}; PRIMS={"INT","FLOAT","STRING","BOOLEAN"}
def is_widget(spec):
    t=spec[0]; cfg=spec[1] if len(spec)>1 else {}
    if isinstance(t,list): return True
    if t=="COMBO" or t in PRIMS: return not cfg.get("forceInput")
    return False
def items(t):
    s=oi[t]["input"]; return [(k,v) for sec in ("required","optional") for k,v in (s.get(sec) or {}).items()]
def wcount(t):
    c=0
    for k,v in items(t):
        if is_widget(v):
            c+=1
            if k in ("seed","noise_seed"): c+=1
    return c
for n in d["nodes"]:
    if n["type"] not in VIRTUAL and n["type"] not in oi: errs.append(f"type {n['type']} unknown")
for n in d["nodes"]:
    if n["type"] in VIRTUAL or n["type"]=="LoadImage": continue
    want,got = wcount(n["type"]), len(n.get("widgets_values") or [])
    if want!=got: errs.append(f"{n['type']} id{n['id']}: widgets_values {got} != {want}")
for l in d["links"]:
    lid,sid,sslot,tid,tslot,_ = l
    if sid not in nodes or tid not in nodes: errs.append(f"link {lid} dangling"); continue
    s,t = nodes[sid], nodes[tid]
    if sslot>=len(s["outputs"]) or tslot>=len(t["inputs"]): errs.append(f"link {lid} bad slot"); continue
    if lid not in (s["outputs"][sslot]["links"] or []): errs.append(f"link {lid} not on src")
    if t["inputs"][tslot]["link"]!=lid: errs.append(f"link {lid} dst mismatch")
    st,tt = s["outputs"][sslot]["type"], t["inputs"][tslot]["type"]
    if s["type"] in VIRTUAL or t["type"] in VIRTUAL: continue
    if st!=tt and tt!="COMBO" and st!="*": errs.append(f"link {lid}: TYPE {s['type']}.{st} -> {t['type']}.{tt}")
for n in d["nodes"]:
    if n["type"] in VIRTUAL: continue
    for k,v in (oi[n["type"]]["input"].get("required") or {}).items():
        if is_widget(v): continue
        inp = next((i for i in n["inputs"] if i["name"]==k), None)
        if inp is None: errs.append(f"{n['type']} id{n['id']}: missing input {k}")
        elif inp["link"] is None: errs.append(f"{n['type']} id{n['id']}: required '{k}' NOT CONNECTED")
# widget VALUE RANGES. Types/links can all be valid while a number sits outside
# its min/max — ComfyUI then refuses the queue with "Input out of range", which
# no amount of link checking catches.
for n in d["nodes"]:
    if n["type"] in VIRTUAL or n["type"] not in oi: continue
    vals = n.get("widgets_values") or []
    vi = 0
    for k, v in items(n["type"]):
        if not is_widget(v): continue
        if vi >= len(vals): break
        val = vals[vi]; vi += 1
        if k in ("seed","noise_seed"): vi += 1
        cfg = v[1] if len(v) > 1 else {}
        if v[0] in ("INT","FLOAT") and isinstance(val,(int,float)) and not isinstance(val,bool):
            lo, hi = cfg.get("min"), cfg.get("max")
            if lo is not None and val < lo:
                errs.append(f"{n['type']} id{n['id']}: {k}={val} BELOW min {lo}")
            if hi is not None and val > hi:
                errs.append(f"{n['type']} id{n['id']}: {k}={val} ABOVE max {hi}")
        if v[0] == "COMBO" or isinstance(v[0], list):
            opts = v[1].get("options") if (len(v)>1 and isinstance(v[1],dict)) else None
            if opts is None and isinstance(v[0], list): opts = v[0]
            if opts and val not in opts:
                errs.append(f"{n['type']} id{n['id']}: {k}={val!r} not a valid option")

sets={n["widgets_values"][0] for n in d["nodes"] if n["type"]=="SetNode"}
for n in d["nodes"]:
    if n["type"]=="GetNode" and n["widgets_values"][0] not in sets: errs.append(f"rail '{n['widgets_values'][0]}' has no Set")
used={n["widgets_values"][0] for n in d["nodes"] if n["type"]=="GetNode"}
for s in sorted(sets-used): warns.append(f"rail '{s}' set but never got")
for n in d["nodes"]:
    if n["type"]=="GetNode" and not (n["outputs"][0]["links"] or []): warns.append(f"Get id{n['id']} feeds nothing")
ids=[n["id"] for n in d["nodes"]]
if len(ids)!=len(set(ids)): errs.append("duplicate node ids")
print(f"nodes={len(d['nodes'])} links={len(d['links'])} groups={len(d['groups'])}")
print(f"ERRORS ({len(errs)}):");   [print("   ",e) for e in errs[:15]]
print(f"WARNINGS ({len(warns)}):"); [print("   ",x) for x in warns[:8]]
