import tiktoken
import os

os.environ['TIKTOKEN_CACHE_DIR'] = ''

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

prompt_381 = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 at the swap meet, 10 were sold to a department store for $25 each. If the remainder were sold for $22 each.
Response: Sold 17 bags for $306, 10 bags for $250. Sold 21 bags for $462. Total sales = $306 + $250 + $462 = $1018. Profit = $1018 - $576 = $442.
User: Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary.
Response: Earns $68 a day, $1360 a month. Annual salary = $1360 * 12 = $16,320.
User: The selling price of a bicycle that sold for $220 last year increased by 15%. What is the new price?
Response: Increment = $220 * 15/100 = $33. New price = $220 + $33 = $253.
User: There are 12 inches to a foot. Blossom's room is 10 feet long in each direction. In square inches, what's the room's area?
Response: Length in inches = 10 feet * 12 inches = 120 inches. Area = 120 * 120 = 14400 square inches.
User: Ruiz's monthly salary is $500. With a 6% raise, what's his new salary?
Response: Raise = $500 x 6/100 = $30. New salary = $500 + $30 = $530.
User: A store sold jeans for $40. 10% discount on every two pairs bought. What's the cost for 3 pairs of jeans?
Response: 2 pairs cost $80, $8 discount. Cost for 2 pairs = $80 - $8 = $72. Total cost = $72 + $40 = $112.
User: A retail store wants to hire 50 phone reps, paid $14.00/hour, 8-hour days. After 5 days, how much will the company pay?
Response: 50 reps * 8 hours/day = 400 hours/day. 400 hours/day * 5 days = 2000 hours. Payment = $14.00/hour * 2000 hours = $28,000.
User: In a factory, 300 employees. 200 earn $12/hr, 40 earn $14/hr. Others earn $17/hr. Cost for one 8-hour shift?
Response: Cost for 200 employees = $2400/hr. Cost for 40 employees = $560/hr. Cost for 60 employees = $1020/hr. Total cost = $2400/hr + $560/hr + $1020/hr = $3980/hr. 8-hour shift = $3980/hr * 8 hours = $31,840.
'''

prompt_731 = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 at the swap meet, 10 were sold to a department store for $25 each. If the remainder were sold for $22 each. How much was the salesman's profit?
Response: The salesman sold 17 bags for 17 bags x $18/bag = $306. He sold 10 more for 10 bags x $25/bag = $250. Thus far he has sold 17 bags + 10 bags = 27 bags. Therefore he sold 48 bags - 27 bags = 21 bags for $22 each. 21 bags were sold for 21 bags x $22/bag = $462. In total the salesman made $306 + $250 + $462 = $1018. The salesman made $1018 - $576 = $442 in profit. Therefore the answer is 442.
User: Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary.
Response: Ursula earns 8 x $8.50 = $68 a day. In a month, she earns 20 x $68 = $1360. Therefore, Ursula's annual salary is 12 x $1360 = $16,320. Therefore the answer is 16320.
User: The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?
Response: The price of the bicycle increased by $220 * 15/100 = $33. Adding the increment price, the new price is $220 + $33 = $253. Therefore the answer is 253.
User: There are 12 inches to a foot. Blossom measures her room and finds that it is exactly 10 feet long in each direction. In square inches, what is the area of her room?
Response: First translate the feet to inches or 10 feet * 12 inches per foot = 120 inches. Then use length * width which is 120 * 120 = 14400 square inches. Therefore the answer is 14400.
User: Ruiz receives a monthly salary of $500. If he received a 6% raise, how much will be Ruiz's new salary?
Response: Ruiz has $500 x 6/100 = $30 salary raise. Thus, his new salary is $500 + $30 = $530. Therefore the answer is 530.
User: A store sold a certain brand of jeans for $40. They only have few pairs of jeans left so they decided to give 10% discount on every two pairs of jeans bought by a customer at any one time. How much will a customer pay for 3 pairs of jeans?
Response: Two pairs of jeans cost $40 x 2 = $80. A discount of $80 x 10/100 = $8 will be given for this. Hence, the cost of two pairs of jeans will now be $80 - $8 = $72. Therefore, the customer will pay $72 + $40 = $112 for the three pairs of jeans. Therefore the answer is 112.
User: A retail store wants to hire 50 new phone reps to assist with the increased call volume that they will experience over the holiday.  Each phone rep will work 8 hours a day and will be paid $14.00 an hour.  After 5 days, how much will the company pay all 50 new employees?
Response: There are 50 reps and they will work 8 hour days so that's 50*8 = 400 hours. They will work 5 days a week so that's 5*400 = 2,000 hours. Each worker will be paid $14.00 an hour and in 5 days they will have worked 2,000 hours so that's 14*2000 = $28,000. Therefore the answer is 28000.
User: In a factory, there are 300 employees. 200 of them earn $12 per hour. Of the rest, 40 of them earn $14 per hour. All others earn $17 per hour. What is the cost to employ all these people for one 8-hour long shift?
Response: The cost of the 200 employees is 200 employees * $12/employee/hour = $2400 per hour. The cost of the 40 employees is 40 employees * $14/employee/hour = $560 per hour. The rest means 300 employees - 200 employees - 40 employees = 60 employees. These 60 employees' employment costs are at 60 employees * $17/employee/hour = $1020
'''

prompt_1368 = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 at the swap meet, 10 were sold to a department store for $25 each. If the remainder were sold for $22 each. How much was the salesman's profit?
Response: The salesman sold 17 bags for 17 bags x $18/bag = $306. He sold 10 more for 10 bags x $25/bag = $250. Thus far he has sold 17 bags + 10 bags = 27 bags. Therefore he sold 48 bags - 27 bags = 21 bags for $22 each. 21 bags were sold for 21 bags x $22/bag = $462. In total the salesman made $306 + $250 + $462 = $1018. The salesman made $1018 - $576 = $442 in profit. Therefore the answer is 442.
User: Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary.
Response: Ursula earns 8 x $8.50 = $68 a day. In a month, she earns 20 x $68 = $1360. Therefore, Ursula's annual salary is 12 x $1360 = $16,320. Therefore the answer is 16320.
User: The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?
Response: The price of the bicycle increased by $220 * 15/100 = $33. Adding the increment price, the new price is $220 + $33 = $253. Therefore the answer is 253.
User: There are 12 inches to a foot. Blossom measures her room and finds that it is exactly 10 feet long in each direction. In square inches, what is the area of her room?
Response: First translate the feet to inches or 10 feet * 12 inches per foot = 120 inches. Then use length * width which is 120 * 120 = 14400 square inches. Therefore the answer is 14400.
User: Ruiz receives a monthly salary of $500. If he received a 6% raise, how much will be Ruiz's new salary?
Response: Ruiz has $500 x 6/100 = $30 salary raise. Thus, his new salary is $500 + $30 = $530. Therefore the answer is 530.
User: A store sold a certain brand of jeans for $40. They only have few pairs of jeans left so they decided to give 10% discount on every two pairs of jeans bought by a customer at any one time. How much will a customer pay for 3 pairs of jeans?
Response: Two pairs of jeans cost $40 x 2 = $80. A discount of $80 x 10/100 = $8 will be given for this. Hence, the cost of two pairs of jeans will now be $80 - $8 = $72. Therefore, the customer will pay $72 + $40 = $112 for the three pairs of jeans. Therefore the answer is 112.
User: A retail store wants to hire 50 new phone reps to assist with the increased call volume that they will experience over the holiday.  Each phone rep will work 8 hours a day and will be paid $14.00 an hour.  After 5 days, how much will the company pay all 50 new employees?
Response: There are 50 reps and they will work 8 hour days so that's 50*8 = 400 hours. They will work 5 days a week so that's 5*400 = 2,000 hours. Each worker will be paid $14.00 an hour and in 5 days they will have worked 2,000 hours so that's 14*2000 = $28,000. Therefore the answer is 28000.
User: In a factory, there are 300 employees. 200 of them earn $12 per hour. Of the rest, 40 of them earn $14 per hour. All others earn $17 per hour. What is the cost to employ all these people for one 8-hour long shift?
Response: The cost of the 200 employees is 200 employees * $12/employee/hour = $2400 per hour. The cost of the 40 employees is 40 employees * $14/employee/hour = $560 per hour. The rest means 300 employees - 200 employees - 40 employees = 60 employees. These 60 employees' employment costs are at 60 employees * $17/employee/hour = $1020 per hour. So in total all employees earn $2400/hour + $560/hour + $1020/hour = $3980/hour. During an 8-hour shift, this cost would be at 8 hours * $3980/hour = $31840. Therefore the answer is 31840.
User: A church has 100 members who've volunteered to bake cookies for the homeless in their local town. If each member baked 10 sheets of cookies, and each sheet has 16 cookies, calculate the total number of cookies the church members baked?
Response: A sheet has 16 cookies, so if each church member baked 10 sheets, they each baked 10 sheets/person * 16 cookies/sheet = 160 cookies/person. The total number of church members who decided to volunteer is 100, so in total, they baked 160 cookies/person * 100 people = 16000 cookies. Therefore the answer is 16000.
User: Sean designs and sells patches.  He orders his patches in a unit of 100 and is charged $1.25 per patch.  If he turns around and sells all 100 patches for $12.00 each, what is his net profit?
Response: He orders 100 patches and they cost $1.25 each so the patches are 100*1.25 = $125.00. He sells each of the 100 patches for $12.00 each so he makes 100*12 = $1,200.00. He makes $1,200.00 selling patches and spends $125.00 to have them made so his net profit is 1200-125 = $1,075.00. Therefore the answer is 1075.
User: A movie theater can hold 50 people at a time.  They charge $8.00 a ticket.  On a Tuesday night they only sold 24 tickets.  By not selling out, how much money did they lose?
Response: The theater can hold 50 people and they charge $8.00 to watch a movie so on a sold-out night they make 50*8 = $400.00. On Tuesday night they only sold 24 tickets at $8.00 apiece so they made 24*8 = $192.00. If they make $400.00 on a sold-out night and they only made $192.00 on Tuesday then they lost 400-192 = $208.00. Therefore the answer is 208.
User: Roger bought a house for $100,000.  He was able to pay 20% down, and his parents paid off an additional 30% of the remaining balance.  How much money does Roger still owe on his house?
Response: 100000(.20)=20000$ down payment. 100000-20000=$80000 remaining. 80000(.30)=24000. 80000-24000=56000$. Therefore the answer is 56000.
User: A bead shop sells one set of crystal beads at $9 each and one set of metal beads at $10 each. Nancy buys one set of crystal beads and two sets of metal beads. How much does she spend in all?
Response: Two sets of metal beads cost 2 x $10 = $20. Therefore, Nancy spends $9 + $20 = $29 in all. Therefore the answer is 29.
User: John buys 3 dress shirts.  They sell for $20 each.  He also has to pay 10% tax on everything.  How much did he pay in total?
Response: The shirts cost 3*$20=$60 before tax. The tax cost $60*.1=$6. So in total they paid $60+$6=$66. Therefore the answer is 66.
User: Mrs. Garcia pays her insurance quarterly at $378. How much does she pay in a year for her insurance?
Response: Since there are three months in a quarter, then there are 12/3 = 4 quarters in a year. So, Mrs. Garcia pays $378 x 4 = $1512 in a year for her insurance. Therefore the answer is 1512.
User: Elizabeth wants to buy the steak knives from a bridal registry as a wedding gift.  The couple registered for 2 steak knife sets and each set contains 4 steak knives.  If Elizabeth buys both sets and they cost $80.00 per set, how much does each single steak knife cost?
Response: Each steak knife set has 4 steak knives and Elizabeth bought 2 sets so there are 4*2 = 8 steak knives in total. Each steak knife set costs $80.00 and she bought 2 sets so she spent $80*2 = $160.00. Both sets cost $160.00 and there are 8 knives in total so each knife costs 160/8 = $20.00 each. Therefore the answer is 20.
'''

prompt_615 = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 for $18 each, 10 for $25 each, and the rest for $22 each. What's his profit?
Response: He sold 17 bags for $306, 10 for $250, and 21 for $462. Total sales: $306 + $250 + $462 = $1018. Profit: $1018 - $576 = $442.
User: Ursula earns $8.50/hour, works 8 hours/day, and 20 days/month. Calculate her annual salary.
Response: Ursula earns $68/day, $1360/month, and $16,320/year.
User: A bicycle sold for $220 last year. Its price increased by 15%. What's the new price?
Response: The price increased by $33, making it $220 + $33 = $253.
User: Blossom's room is 10 feet long. What's the room's area in square inches?
Response: Room length in inches: 10 * 12 = 120 inches. Area: 120 * 120 = 14,400 square inches.
User: Ruiz gets a $30 raise on his $500 monthly salary. What's his new salary?
Response: New salary: $500 + $30 = $530.
User: A store sells jeans for $40. They offer a 10% discount on every two pairs bought. What's the cost of 3 pairs?
Response: Two pairs cost $80, with a $8 discount, making it $72 each. Total cost: $72 + $40 = $112.
User: A company hires 50 reps for 8 hours/day at $14/hour. Calculate their pay for 5 days.
Response: Total hours: 50 reps * 8 hours/day * 5 days = 2,000 hours. Total pay: 2,000 hours * $14/hour = $28,000.
User: In a factory, 200 employees earn $12/hour, 40 earn $14/hour, and the rest earn $17/hour. What's the cost for an 8-hour shift?
Response: Cost for 200 employees: 200 * $12 * 8 = $19,200. Cost for 40 employees: 40 * $14 * 8 = $4,480. Cost for 60 employees: 60 * $17 * 8 = $81,600. Total cost: $19,200 + $4,480 + $81,600 = $105,280.
User: A church's 100 members each bake 10 sheets of cookies with 16 cookies/sheet. How many cookies did they bake?
Response: Total cookies: 100 members * 10 sheets/member * 16 cookies/sheet = 16,000 cookies.
User: Sean orders 100 patches for $1.25 each, sells them for $12 each. What's his net profit?
Response: Cost: 100 patches * $1.25/patch = $125. Revenue: 100 patches * $12/patch = $1,200. Net profit: $1,200 - $125 = $1,075.
User: A theater holds 50 people, charging $8/ticket. They sold 24 tickets on Tuesday. How much money did they lose?
Response: Maximum revenue: 50 * $8 = $400. Tuesday revenue: 24 * $8 = $192. Loss: $400 - $192 = $208.
User: Roger bought a $100,000 house, paid 20% down, and his parents paid 30% of the remaining balance. How much does he owe now?
Response: Down payment: $100,000 * 20% = $20,000. Remaining balance: $100,000 - $20,000 = $80,000. Parent's payment: $80,000 * 30% = $24,000. Roger owes: $80,000 - $24,000 = $56,000.
User: A bead shop sells crystal beads for $9 each and metal beads for $10 each. Nancy buys 1 crystal set and 2 metal sets. How much does she spend?
Response: Nancy spends $9 + 2 * $10 = $29.
User: John buys 3 dress shirts for $20 each, with 10% tax. What's his total cost?
Response: Shirts cost $60 before tax. Tax: $60 * 10% = $6. Total cost: $60 + $6 = $66.
User: Mrs. Garcia pays $378 quarterly for insurance. What's her annual payment?
Response: Quarterly payment: $378. Annual payment: $378 * 4 = $1,512.
User: Elizabeth buys 2 sets of steak knives for $80 each, with 4 knives in each set. What's the cost per knife?
Response: She buys 8 knives for $160, making it $160/8 = $20 per knife.
'''

prompt_557 = '''
User: A salesman bought 48 backpacks for $576. He sold 17 for $18 each, 10 for $25 each, and the rest for $22 each. What's his profit?
Response: The salesman sold 17 bags for $306, 10 for $250, and 21 for $462. His total sales are $1018. The profit is $442.
User: Ursula earns $8.50 an hour, working 8 hours a day for 20 days a month. What's her annual salary?
Response: Ursula earns $68 a day, which is $1360 a month. Her annual salary is $16,320.
User: The selling price of a bicycle that was $220 increased by 15%. What's the new price?
Response: The price increased by $33, so the new price is $253.
User: Blossom's room is 10 feet long in each direction. What's the area in square inches?
Response: Her room is 120 inches long. The area is 14,400 square inches.
User: Ruiz received a $30 raise on his $500 monthly salary. What's his new salary?
Response: Ruiz's new salary is $530.
User: A store sells jeans for $40 each and offers a 10% discount for every two pairs bought together. How much for 3 pairs?
Response: Two pairs cost $72. Three pairs cost $112.
User: A retail store wants to hire 50 phone reps working 8 hours a day for $14 per hour. After 5 days, how much will they pay?
Response: They will work 2,000 hours in total. The cost is $28,000.
User: In a factory, 200 employees earn $12/hour, 40 earn $14/hour, and the rest earn $17/hour. How much for one 8-hour shift?
Response: The cost for 200 employees is $2,400 per hour, for 40 employees it's $560 per hour, and for 60 employees it's $1,020 per hour. The total is $3,980 per hour, or $31,840 for an 8-hour shift.
User: A church's 100 members baked 16,000 cookies. Each baked 160 cookies. What's the total?
Response: Each member baked 160 cookies, totaling 16,000 cookies.
User: Sean buys 100 patches at $1.25 each and sells them for $12 each. What's his net profit?
Response: Sean spent $125 and earned $1,200, so his net profit is $1,075.
User: A theater can hold 50 people at $8 per ticket. They sold 24 tickets. How much money did they lose?
Response: They made $192, so they lost $208.
User: Roger bought a $100,000 house with a 20% down payment and his parents paid 30% of the remaining balance. What's his remaining debt?
Response: Roger paid $20,000 down, leaving a $80,000 balance. His parents paid $24,000, so his remaining debt is $56,000.
User: A bead shop sells crystal beads for $9 each and metal beads for $10 each. Nancy buys 1 set of crystal beads and 2 sets of metal beads. How much does she spend?
Response: Nancy spent $9 for crystal beads and $20 for metal beads. In total, she spent $29.
User: John buys 3 dress shirts at $20 each with 10% tax. How much did he pay?
Response: The shirts cost $60 before tax and $6 in tax, so he paid $66.
User: Mrs. Garcia pays her $378 insurance quarterly. What's her annual cost?
Response: She pays $1,512 annually.
User: Elizabeth buys 2 sets of steak knives for $80 each, each set containing 4 knives. What's the cost per knife?
Response: Elizabeth bought 8 knives for $160, so each knife costs $20.
'''

prompt_1643 = '''User: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Response: Let's think step by step
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

User: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Response: Let's think step by step
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

User: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
Response: Let's think step by step
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

User: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
Response: Let's think step by step
For the first three baskets, the number of apples and oranges in one basket is 9+15=24
In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.
Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.
The number of apples in the fourth basket is 9-2=7
There are also 15-2=13 oranges in the fourth basket
The combined number of oranges and apples in the fourth basket is 13+7=20
The fourth basket also contains 14-2=12 bananas.
In total, the fourth basket has 20+12=32 fruits.
The four baskets together have 32+114=146 fruits.
The answer is 146

User: You can buy 4 apples or 1 watermelon for the same price. You bought 36 fruits evenly split between oranges, apples and watermelons, and the price of 1 orange is $0.50. How much does 1 apple cost if your total bill was $66?
Response: Let's think step by step
If 36 fruits were evenly split between 3 types of fruits, then I bought 36/3 = 12 units of each fruit
If 1 orange costs $0.50 then 12 oranges will cost $0.50 * 12 = $6
If my total bill was $66 and I spent $6 on oranges then I spent $66 - $6 = $60 on the other 2 fruit types.
Assuming the price of watermelon is W, and knowing that you can buy 4 apples for the same price and that the price of one apple is A, then 1W=4A
If we know we bought 12 watermelons and 12 apples for $60, then we know that $60 = 12W + 12A
Knowing that 1W=4A, then we can convert the above to $60 = 12(4A) + 12A
$60 = 48A + 12A
$60 = 60A
Then we know the price of one apple (A) is $60/60= $1
The answer is 1

User: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students.  At the start of the school year, Susy had 100 social media followers.  She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week.  Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week.  After three weeks, how many social media followers did the girl with the most total followers have?
Response: Let's think step by step
After one week, Susy has 100+40 = 140 followers.
In the second week, Susy gains 40/2 = 20 new followers.
In the third week, Susy gains 20/2 = 10 new followers.
In total, Susy finishes the three weeks with 140+20+10 = 170 total followers.
After one week, Sarah has 50+90 = 140 followers.
After the second week, Sarah gains 90/3 = 30 followers.
After the third week, Sarah gains 30/3 = 10 followers.
So, Sarah finishes the three weeks with 140+30+10 = 180 total followers.
Thus, Sarah is the girl with the most total followers with a total of 180.
The answer is 180

User: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in dollars?
Response: Let's think step by step
Sam bought 12 boxes x $10 = $120 worth of highlighters.
He bought 12 * 30 = 360 highlighters in total.
Sam then took 5 boxes * 6 highlighters/box = 30 highlighters.
He sold these boxes for 5 * $3 = $15
After selling these 5 boxes there were 360 - 30 = 330 highlighters remaining.
These form 330 / 3 = 110 groups of three pens.
He sold each of these groups for $2 each, so made 110 * 2 = $220 from them.
In total, then, he earned $220 + $15 = $235.
Since his original cost was $120, he earned $235 - $120 = $115 in profit.
The answer is 115

User: In a certain school, 2/3 of the male students like to play basketball, but only 1/5 of the female students like to play basketball. What percent of the population of the school do not like to play basketball if the ratio of the male to female students is 3:2 and there are 1000 students?
Response: Let's think step by step
The students are divided into 3 + 2 = 5 parts where 3 parts are for males and 2 parts are for females.
Each part represents 1000/5 = 200 students.
So, there are 3 x 200 = 600 males.
And there are 2 x 200 = 400 females.
Hence, 600 x 2/3 = 400 males play basketball.
And 400 x 1/5 = 80 females play basketball.
A total of 400 + 80 = 480 students play basketball.
Therefore, 1000 - 480 = 520 do not like to play basketball.
The percentage of the school that do not like to play basketball is 520/1000 * 100 = 52
The answer is 52
'''

boolq = '''
User: Passage: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet. Question: do iran and afghanistan speak the same language
Response: True

User: Passage: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable. Question: do good samaritan laws protect those who help at an accident
Response: True

User: Passage: Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, and Flickr. Question: is windows movie maker part of windows essentials
Response: True

User: Passage: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective. Question: is elder scrolls online the same as skyrim
Response: False
'''

prompt_half = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 at the swap meet, 10 were sold to a department store for $25 each. If the remainder were sold for $22 each. How much was the salesman's profit?
Response: The salesman sold 17 bags for 17 bags x $18/bag = $306. He sold 10 more for 10 bags x $25/bag = $250. Thus far he has sold 17 bags + 10 bags = 27 bags. Therefore he sold 48 bags - 27 bags = 21 bags for $22 each. 21 bags were sold for 21 bags x $22/bag = $462. In total the salesman made $306 + $250 + $462 = $1018. The salesman made $1018 - $576 = $442 in profit. Therefore the answer is 442.
User: Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary.
Response: Ursula earns 8 x $8.50 = $68 a day. In a month, she earns 20 x $68 = $1360. Therefore, Ursula's annual salary is 12 x $1360 = $16,320. Therefore the answer is 16320.
User: The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?
Response: The price of the bicycle increased by $220 * 15/100 = $33. Adding the increment price, the new price is $220 + $33 = $253. Therefore the answer is 253.
User: There are 12 inches to a foot. Blossom measures her room and finds that it is exactly 10 feet long in each direction. In square inches, what is the area of her room?
Response: First translate the feet to inches or 10 feet * 12 inches per foot = 120 inches. Then use length * width which is 120 * 120 = 14400 square inches. Therefore the answer is 14400.
User: Ruiz receives a monthly salary of $500. If he received a 6% raise, how much will be Ruiz's new salary?
Response: Ruiz has $500 x 6/100 = $30 salary raise. Thus, his new salary is $500 + $30 = $530. Therefore the answer is 530.
User: A store sold a certain brand of jeans for $40. They only have few pairs of jeans left so they decided to give 10% discount on every two pairs of jeans bought by a customer at any one time. How much will a customer pay for 3 pairs of jeans?
Response: Two pairs of jeans cost $40 x 2 = $80. A discount of $80 x 10/100 = $8 will be given for this. Hence, the cost of two pairs of jeans will now be $80 - $8 = $72. Therefore, the customer will pay $72 + $40 = $112 for the three pairs of jeans. Therefore the answer is 112.
User: A retail store wants to hire 50 new phone reps to assist with the increased call volume that they will experience over the holiday.  Each phone rep will work 8 hours a day and will be paid $14.00 an hour.  After 5 days, how much will the company pay all 50 new employees?
Response: There are 50 reps and they will work 8 hour days so that's 50*8 = 400 hours. They will work 5 days a week so that's 5*400 = 2,000 hours. Each worker will be paid $14.00 an hour and in 5 days they will have worked 2,000 hours so that's 14*2000 = $28,000. Therefore the answer is 28000.
User: In a factory, there are 300 employees. 200 of them earn $12 per hour. Of the rest, 40 of them earn $14 per hour. All others earn $17 per hour. What is the cost to employ all these people for one 8-hour long shift?
Response: The cost of the 200 employees is 200 employees * $12/employee/hour = $2400 per hour. The cost of the 40 employees is 40 employees * $14/employee/hour = $560 per hour. The rest means 300 employees - 200 employees - 40 employees = 60 employees. These 60 employees' employment costs are at 60 employees * $17/employee/hour = $1020 per hour. So in total all employees earn $2400/hour + $560/hour + $1020/hour = $3980/hour. During an 8-hour shift, this cost would be at 8 hours * $3980/hour = $31840. Therefore the answer is 31840.
'''
prompts = [
    # prompt_381,
    # prompt_731,
    # prompt_1368,
    # prompt_615,
    # prompt_557,
    # prompt_1643,
    # boolq,
    prompt_half
]

for prompt in prompts:
    print(num_tokens_from_string(prompt, "gpt-3.5-turbo"))