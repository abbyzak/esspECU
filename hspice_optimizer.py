import subprocess
import os
import re


# config - change this if hspice comand is diffrnt
HSPICE_CMD = 'hspice'  # might be 'hspicerf' or full path


def gennetlst(fan, n):
    # bse template
    hdr = """Lab 1 Problem 1A

* Bring in the library ... 
.lib 'cmoslibrary.lib' nominal

* My VCC is 
.param pvcc = 3

* Sizing Variables
.param alpha = 1.7

* Set Power and Ground as Global
.global vcc! gnd!

.subckt inv A Z 
  m1 Z A gnd! gnd! nmos w=1.4u l=0.35u AD=0.7p 
  m2 Z A vcc! vcc! pmos w=(1.4u*alpha) l=0.35u AD=0.7p*alpha  
.ends 

Cload z gnd! 30pF

Vin a gnd! 0V PWL 0 0NS 1NS 3 20NS 3

* Power Supplies
Vgnd gnd! 0 DC = 0
Vvcc vcc! 0 DC = 3V

* Analysis
.tran 1NS 200NS
.print tran v(a) v(z)

.OPTION MEASFORM=3

.OPTION POST
.TEMP 25 

.measure TRAN tphl_inv  TRIG v(Xinv1.a) VAL = 1.5 RISE = 1 TARG v(z) VAL=1.5 FALL = 1

.param fan = """
    
    hdr += str(fan) + "\n"
    
    # add invrtrs
    invs = ""
    nds = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
    
    for idx in range(n):
        if idx == n-1:
            invs += f"Xinv{idx+1} {nds[idx]} z inv M=fan**{idx}\n"
        else:
            invs += f"Xinv{idx+1} {nds[idx]} {nds[idx+1]} inv M=fan**{idx}\n"
    
    invs += ".end\n"
    
    fll = hdr + invs


    with open('InvChain.sp','w') as f:
        f.write(fll)


def runhspce():
    # run simultn
    try:
        result = subprocess.run([HSPICE_CMD,'InvChain.sp'], 
                              capture_output=True, text=True, timeout=60)
        
        return result.returncode == 0
    except FileNotFoundError:
        print(f"\nERROR: '{HSPICE_CMD}' not found!")
        print("You need to run this on eecad server where HSPICE is installed")
        print("Or update HSPICE_CMD variable at top of script")
        return False
    except Exception as e:
        print(f"Exceptn: {str(e)[:50]}")
        return False


def getdly():
    # try both .mt0 and .lis files
    fnames = ['InvChain.mt0', 'InvChain.lis']
    
    for fname in fnames:
        try:
            if not os.path.exists(fname):
                continue
                
            with open(fname,'r') as f:
                cntnt = f.read()
            
            # look for measurment    
            mtch = re.search(r'tphl_inv\s*=\s*([\d\.eE\+\-]+)', cntnt)
            if mtch:
                return float(mtch.group(1))
        except:
            pass
    
    return None


# check if hspice availble
print("Checkng HSPICE availabilty...")
try:
    result = subprocess.run([HSPICE_CMD, '-v'], 
                          capture_output=True, text=True, timeout=5)
    print("HSPICE found!\n")
except:
    print(f"\nWARNING: Cannot find '{HSPICE_CMD}'")
    print("Make sure you're running this on eecad server\n")
    resp = input("Continue anyway? (y/n): ")
    if resp.lower() != 'y':
        exit(1)

print("\nStrtng HSPICE optimiztion...")
print("=" * 50)

bstfan = None
bstn = None
bstdly = float('inf')


# try diffrnt combos
fanvals = [2,3,4,5,6,7,8]
nvals = [3,4,5,6,7,8,9,10,11,12]

for fan in fanvals:
    for n in nvals:
        print(f"\nTrying fan={fan}, N={n}...", end=" ")
        
        gennetlst(fan, n)
        
        if runhspce():
            dly = getdly()
            
            if dly is not None:
                print(f"Delay = {dly:.6e} s")
                
                if dly < bstdly:
                    bstdly = dly
                    bstfan = fan
                    bstn = n
            else:
                print("Culdnt extact delay")
        else:
            print("Simultion faild")
            break  # stop if first one fails
    
    if bstfan is None and fan == fanvals[0] and n == nvals[-1]:
        print("\nAll simulatns failng. Check HSPICE setup.")
        break


print("\n" + "=" * 50)
print("OPTIMIZTION COMPLETE")
print("=" * 50)

if bstfan is not None:
    print(f"\nOptimal confguration:")
    print(f"  Fan-out (fan) = {bstfan}")
    print(f"  Number of invertrs (N) = {bstn}")
    print(f"  Minimum delay = {bstdly:.6e} seconds")
    print(f"\nThe optimal chain has {bstn} inverters")
else:
    print("\nNo valid rsults found!")
    print("Make sure you run this on eecad server where HSPICE is installed")
