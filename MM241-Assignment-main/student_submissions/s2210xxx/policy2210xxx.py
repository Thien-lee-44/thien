from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self):
        # Student code here
        pass
    def ___stock_area___(self,stock):
        x,y=self._get_stock_size_(stock)
        return x*y
    def ___max_size___(self,stock):
        return  np.max(np.sum(stock!=-2, axis=1)),np.max(np.sum(stock!=-2, axis=1))
    def get_action(self, observation, info):
            list_prods =sorted(observation["products"],key=lambda x: x["size"][0]*x["size"][1],reverse=True)
            list_stock = sorted(enumerate(observation["stocks"]),key=lambda x:self.___stock_area___(x[1]),reverse=True)
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = None, None

            # Pick a product that has quality > 0
            
            for i, stock in list_stock:
                product=-1
                dS=-1
                mx,my=self.___max_size___(stock)
                S=mx*my
                stock_w, stock_h = self._get_stock_size_(stock)
                print(self._get_stock_size_(stock),i)
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        prod_size=prod["size"]
                        prod_w, prod_h =   prod_size
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        cx=None
                        cy=None
                        prod_w, prod_h = prod_size
                        for x in range(stock_w - prod_w + 1):
                            if x>=mx:
                                continue
                            for y in range(stock_h - prod_h + 1):
                                if y>=my:
                                    continue
                                if self._can_place_(stock, (x, y), prod_size):
                                    if (x+prod_w)*(y+prod_h)-S<dS or dS==-1:
                                        cx=x
                                        cy=y
                                        dS=(x+prod_w)*(y+prod_h)-S
                        pos_x,pos_y=cx,cy
                        if pos_x is not None and pos_y is not None:
                            break

                if pos_x is not None and pos_y is not None:
                    stock_idx = i
                    break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    # Student code here
    # You can add more functions if needed
