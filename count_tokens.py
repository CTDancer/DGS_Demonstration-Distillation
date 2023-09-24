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
User: Passage: Captain America: Civil War is a 2016 American superhero film based on the Marvel Comics character Captain America. It is the sequel to 2011's Captain America: The First Avenger and 2014's Captain America: The Winter Soldier. Question: is captain america civil war an avengers movie?
Response: True

User: Passage: Jockeys must be light to ride at the weights assigned to their mounts. The weight of a jockey usually ranges from 108 to 118 lb. Question: do jockeys have to weigh a certain amount?
Response: True

User: Passage: In 2014, The Hollywood Reporter stated that the seventh Tinkerbell film was canceled. Question: will there be another tinkerbell movie after neverbeast?
Response: False

User: Passage: Francis E. Warren Air Force Base is located approximately 3 miles west of Cheyenne, Wyoming. Question: is there an air force base in Wyoming?
Response: True

User: Passage: In the mid-season finale "Pretty Much Dead Already," Sophia is found as a walker and put down. Question: do they find Sophia on the walking dead?
Response: True

User: Passage: The Equinox and the Torrent are larger than the Vue. Question: is the Saturn Vue the same as the Chevy Equinox?
Response: False

User: Passage: The Sinner is based on the novel of the same name by German crime writer Petra Hammesfahr. Question: is The Sinner on Netflix based on a book?
Response: True

User: Passage: No NFL team has yet played a Super Bowl in their own stadium. Question: has an NFL team ever played a Super Bowl in their own stadium?
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

prompt_quater = '''
User: A salesman bought a case of 48 backpacks for $576. He sold 17 of them for $18 at the swap meet, 10 were sold to a department store for $25 each. If the remainder were sold for $22 each. How much was the salesman's profit?
Response: The salesman sold 17 bags for 17 bags x $18/bag = $306. He sold 10 more for 10 bags x $25/bag = $250. Thus far he has sold 17 bags + 10 bags = 27 bags. Therefore he sold 48 bags - 27 bags = 21 bags for $22 each. 21 bags were sold for 21 bags x $22/bag = $462. In total the salesman made $306 + $250 + $462 = $1018. The salesman made $1018 - $576 = $442 in profit. Therefore the answer is 442.
User: Ursula earns $8.50 an hour working in a restaurant. She works 8 hours a day. If she works 20 days a month, determine her annual salary.
Response: Ursula earns 8 x $8.50 = $68 a day. In a month, she earns 20 x $68 = $1360. Therefore, Ursula's annual salary is 12 x $1360 = $16,320. Therefore the answer is 16320.
User: The selling price of a bicycle that had sold for $220 last year was increased by 15%. What is the new price?
Response: The price of the bicycle increased by $220 * 15/100 = $33. Adding the increment price, the new price is $220 + $33 = $253. Therefore the answer is 253.
User: There are 12 inches to a foot. Blossom measures her room and finds that it is exactly 10 feet long in each direction. In square inches, what is the area of her room?
Response: First translate the feet to inches or 10 feet * 12 inches per foot = 120 inches. Then use length * width which is 120 * 120 = 14400 square inches. Therefore the answer is 14400.
User: A retail store wants to hire 50 new phone reps to assist with the increased call volume that they will experience over the holiday.  Each phone rep will work 8 hours a day and will be paid $14.00 an hour.  After 5 days, how much will the company pay all 50 new employees?
Response: There are 50 reps and they will work 8 hour days so that's 50*8 = 400 hours. They will work 5 days a week so that's 5*400 = 2,000 hours. Each worker will be paid $14.00 an hour and in 5 days they will have worked 2,000 hours so that's 14*2000 = $28,000. Therefore the answer is 28000.
User: In a factory, there are 300 employees. 200 of them earn $12 per hour. Of the rest, 40 of them earn $14 per hour. All others earn $17 per hour. What is the cost to employ all these people for one 8-hour long shift?
Response: The cost of the 200 employees is 200 employees * $12/employee/hour = $2400 per hour. The cost of the 40 employees is 40 employees * $14/employee/hour = $560 per hour. The rest means 300 employees - 200 employees - 40 employees = 60 employees. These 60 employees' employment costs are at 60 employees * $17/employee/hour = $1020 per hour. So in total all employees earn $2400/hour + $560/hour + $1020/hour = $3980/hour. During an 8-hour shift, this cost would be at 8 hours * $3980/hour = $31840. Therefore the answer is 31840.
'''

prompt_1126 = '''
User: Janet hires six employees. Four are warehouse workers making $15/hour, and two are managers making $20/hour. Janet pays 10% in FICA taxes. If everyone works 25 days a month and 8 hours a day, what's the total cost for their wages and taxes for one month?
Response: To calculate the total cost, first, find each worker's monthly wage. Warehouse workers earn $15/hour * 8 hours/day * 25 days/month = $3,000/month. Managers earn $20/hour * 8 hours/day * 25 days/month = $4,000/month. The total wages are $3,000 (workers) + $4,000 (managers) = $7,000. Now, calculate the FICA tax, which is 10% of the total wages: $7,000 * 0.10 = $700. The grand total is $7,000 (wages) + $700 (taxes) = $7,700.
User: Sam works at the Widget Factory. He assembles 1 widget every 10 minutes. Jack helps and they assemble 2 widgets every 15 minutes. Tony worked a full 8-hour shift. They completed 68 widgets. How long does it take Tony to assemble one widget?
Response: Sam takes 10 minutes to assemble 1 widget. Jack takes the same time, so they both take 10 minutes per widget. Tony produced the remaining 68 - 36 (Sam) - 16 (Jack) = 16 widgets. Tony took 8 hours to complete those 16 widgets, so he takes 8 hours * 60 minutes/hour / 16 widgets = 30 minutes per widget.
User: Sarah and Ellie got lost in a house of mirrors. Sarah saw her reflection 10 times in tall mirrors and 5 times in wide mirrors. Ellie saw her reflection 6 times in tall mirrors and 3 times in wide mirrors. They each passed tall mirrors 3 times and wide mirrors 5 times. How many times did they see their reflections in total?
Response: Sarah saw her reflection 10 times (tall mirrors) + 5 times (wide mirrors) = 15 times in one pass. Ellie saw her reflection 6 times (tall mirrors) + 3 times (wide mirrors) = 9 times in one pass. Together, they saw their reflections 15 times (Sarah) + 9 times (Ellie) = 24 times in one pass. They passed tall mirrors 3 times and wide mirrors 5 times, so the total reflections they saw are 24 times (one pass) * 3 (passes) + 24 times (one pass) * 5 (passes) = 72 times + 120 times = 192 times in total.
User: The curry house used to buy peppers for 30 very spicy curries, 30 spicy curries, and 10 mild curries. Now they buy for 15 spicy curries and 90 mild curries. They no longer sell very spicy curries. How many fewer peppers do they buy now?
Response: Previously, the curry house bought 3 peppers per very spicy curry * 30 very spicy curries + 2 peppers per spicy curry * 30 spicy curries + 1 pepper per mild curry * 10 mild curries = 90 + 60 + 10 = 160 peppers. Now, they buy 2 peppers per spicy curry * 15 spicy curries + 1 pepper per mild curry * 90 mild curries = 30 + 90 = 120 peppers. The difference is 160 peppers (before) - 120 peppers (now) = 40 fewer peppers.
User: Hasan filled a box with 38 dinner plates. The scale reported it was too heavy. He removed plates until it was acceptable. Each plate weighs 10 ounces, and the scale's limit is 20 pounds. How many plates did he need to remove?
Response: Hasan had to remove plates until the box was under the 20-pound limit. Each plate weighs 10 ounces, which is 10/16 pounds. So, the original weight was 38 plates * 10/16 pounds per plate = 23.75 pounds. He needed to remove enough plates to get the weight under 20 pounds. The difference is 23.75 pounds (original) - 20 pounds (limit) = 3.75 pounds. To find out how many plates to remove, divide 3.75 pounds / (10/16 pounds per plate) = 6 plates.
User: Alec wants 3/4 of the class to vote for him to win Class President. Half the class already supports him, and 5 are considering it. He surveys others, makes changes, and now has a fifth of them considering. The class has 60 students. How many more votes does he need?
Response: Alec's goal is to get 3/4 of the class to vote for him, which is 3/4 * 60 students = 45 votes. He already has half the class, which is 60 students / 2 = 30 votes. Additionally, 5 students are considering it, so that's 30 votes (half) + 5 votes (considering) = 35 votes. Alec needs 45 votes (goal) - 35 votes (current) = 10 more votes to reach his goal.
'''

distill_prompt = '''
I'm giving you several User-Response pairs, delimited by triple backticks.
```{previous_demos}```

# Task:
1. Distill the given User-Response pairs to be succinct while keeping the response logic and format and satisfying all the requirements.
2. After distillation, don't rush to give your result. Examine each User-Response pair and check whether each pair satisifies \
all the requirements. If not, you should modify your result accordingly.
3. Finally you can give your distillation result.

# Requirements:
1. In each User message, besides all the questions, preserve all the information related to these questions and the Response message \
and then omit other unnecessary information.
2. For each Response message, if there is a step-by-step derivation to the final answer in the initial version, you must preserve it INTACT in your distillation result.
3. Must NOT change or omit the final answers in each Response message.
4. Must NOT change or omit questions in each User message.
5. If the User-Response pair in the initial version has a step-by-step derivation to the final answer in the Response message, \
then you must also present this step-by-step derivation explicitly in the Response message in your distillation result.
6. The format of User messages and Response messages in your result must be the same as in the given version.

# Note:
If you think a User-Response pair does not need distillation, you should keep it intact instead of omitting it. \
Thus, the number of User-Response pairs in your distillation result should be the same as the given User-Response pairs.
'''
prompt_5 = '''
User: Janet hires six employees. Four are warehouse workers making $15/hour, and two are managers making $20/hour. Janet pays 10% of their salaries in FICA taxes. If everyone works 25 days a month, 8 hours a day, how much does Janet owe for wages and taxes in one month?
Response: To calculate, we start with each worker's monthly wage. Warehouse workers make $15/hour * 8 hours/day * 25 days = $3000/month. Managers make $20/hour * 8 hours/day * 25 days = $4000/month. In total, warehouse workers cost $12,000, and managers cost $8,000. Adding them gives $20,000 for wages. FICA taxes are 10% of this, so taxes amount to $2,000. Thus, the total is $22,000.
User: Sam assembles 1 widget every 10 minutes. Jack helps and they finish 2 widgets every 15 minutes. Tony joined recently. Sam worked 6 hours, Jack 4 hours, Tony 8 hours, completing 68 widgets. How long for Tony to assemble 1 widget (in minutes)?
Response: Sam takes 10 minutes/widget. Jack helps in 15 minutes, completing 2 widgets, so Jack takes 7.5 minutes/widget. Tony took 8 hours to complete 16 widgets, so he takes 30 minutes/widget.
User: Sarah saw her reflection 10 times in tall mirrors and 5 times in wide mirrors. Ellie saw her reflection 6 times in tall mirrors and 3 times in wide mirrors. They passed tall mirrors 3 times and wide mirrors 5 times each. How many times did they see their reflections?
Response: Sarah saw her reflection 30 times in tall mirrors and 25 times in wide mirrors. Ellie saw her reflection 18 times in tall mirrors and 15 times in wide mirrors. Together, they saw their reflections 88 times.
User: The curry house used to buy peppers for 30 very spicy curries, 30 spicy curries, and 10 mild curries. Now they buy for 15 spicy curries and 90 mild curries. No more very spicy curries. How many fewer peppers do they buy?
Response: Originally, they bought 160 peppers. Now, they buy 120 peppers. They now buy 40 fewer peppers.
User: Hasan packed plates in a box until it was too heavy, then removed plates. Each plate weighs 10 ounces, and the limit is 20 pounds. How many plates did Hasan remove?
Response: Hasan removed 6 plates. The weight limit is 320 ounces. Initially, the box had 380 ounces of plates. Removing 6 plates left 320 ounces, which is within the limit.
'''

prompt_3 = '''
User: Alec is running for Class President. He thinks that if he can get three-quarters of the class to vote for him then there is no chance anyone else can beat him. Half of the class have already said they will vote for him but out of the remaining students, only 5 have said they are thinking about voting for him. He surveys the students who are thinking about voting for someone else, and changes his flyers to reflect the issues these students are concerned about. This results in a fifth of these students saying they'll vote for him. If Alec's class has 60 students and everyone who said they will vote for him does so, how many more votes does Alec need to reach his goal number of votes?
Response: Alec's goal is to get three-quarters of the class to vote for him, which is 60 students * 3/4 = 45 votes. Currently, he has 30 votes from half the class and an additional 5 students considering voting for him. So, he has 30 + 5 = 35 votes. To reach his goal, he needs 45 - 35 = 10 more votes.
User: Peggy is moving and is looking to get rid of her record collection. Sammy says that he will buy all of them for 4 dollars each. Bryan is only interested in half of the records but will offer 6 dollars each for the half that he is interested in and 1 dollar each for the remaining half that he is not interested in with the hopes that he can resell them in bulk later. If Peggy has 200 records, what is the difference in profit between Sammy versus Bryan's deal?
Response: Sammy offers $4 each for all 200 records, totaling $4 * 200 = $800. Bryan offers $6 each for half (100 records * $6 = $600) and $1 each for the other half (100 records * $1 = $100). The difference in profit between Sammy and Bryan's deals is $800 - ($600 + $100) = $100.
User: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Response: Angelo and Melanie have 2 chapters, each requiring 3 hours (2 * 3 = 6 hours), and 4 worksheets, each needing 1.5 hours (4 * 1.5 = 6 hours). In total, they need to dedicate 6 + 6 = 12 hours to their study materials. They plan to study no more than 4 hours per day. Accounting for breaks, including 10-minute breaks every hour, 3 10-minute snack breaks, and 30 minutes for lunch, they need an additional 3 hours each day. Thus, they need 12 hours + 3 hours of breaks = 15 hours per day. Over the next week, they should plan to study for 15 hours / 4 hours per day = 3.75 days, which they should round up to 4 days.
'''

prompt_1214 = '''
User: Jeremie wants to go to an amusement park with 3 friends at the end of summer. Tickets are $18 for children and a set of snack cost $5. How much will it cost Jeremie and her 3 friends to go to the amusement park and buy each of them a set of snacks?
Response: 4 people will go to the park. Tickets are $18 each, and snacks cost $5 per set. So, it's 4 x ($18 + $5) = $92 in total.

User: Summer and Jolly both went to the same middle school. However, when they finished college, Summer had five more degrees than Jolly. If Summer has a total of 150 degrees, what's the combined number of degrees they both have?
Response: Jolly has 150 - 5 = 145 degrees. Combined, they have 150 + 145 = 295 degrees.

User: Uncle Bradley has a $1000 bill that he wants to change into smaller bills so he could give them to his nieces and nephews. He wants to change 3/10 of the money into $50 bills while the rest into $100 bills. How many pieces of bills will Uncle Bradley have in all?
Response: $300 will be in $50 bills (3/10 x $1000), which is 6 bills. The remaining $700 will be in $100 bills, which is 7 bills. In total, he'll have 6 + 7 = 13 bills.

User: Leo had 400 marbles in a jar. He gave Manny 1/4 of the number of packs of marbles, and Neil received 1/8 of the number of packs of marbles. How many packs of marbles did Leo keep?
Response: Leo had 40 packs of marbles. He gave Manny 10 packs and Neil 5 packs. So, he kept 40 - 10 - 5 = 25 packs.

User: Joy has 30 pencils, and Colleen has 50 pencils. If they bought the pencils at $4 each, how much more money did Colleen pay than Joy for her pencils?
Response: Colleen has 20 more pencils than Joy, so she paid $4 x 20 = $80 more.

User: Angus, Patrick, and Ollie went fishing for trout on Rainbow Lake. Angus caught 4 more fish than Patrick did, but Ollie caught 7 fewer fish than Angus. If Ollie caught 5 fish, how many fish did Patrick catch?
Response: Angus caught 12 fish, and Patrick caught 8 fish.

User: It takes 5 people to lift a car and twice as many people to lift a truck. How many people are needed to lift 6 cars and 3 trucks?
Response: It takes 10 people to lift a truck. To lift 6 cars, you'd need 5 x 6 = 30 people, and to lift 3 trucks, you'd need 10 x 3 = 30 people. In total, 30 + 30 = 60 people are needed.

User: Last month, a factory made 12000 dolls and their associated accessories. The accessories for each doll included 2 shoes, 3 bags, 1 set of cosmetics, and 5 hats. If each doll took 45 seconds to make and each accessory took 10 seconds to make, what was the total combined machine operation time, in seconds, required to manufacture all of the dolls and their accessories?
Response: The factory made 24000 shoes, 36000 bags, 12000 cosmetics sets, and 60000 hats. Dolls took 540000 seconds, and accessories took 1320000 seconds to make. So, in total, it took 1860000 seconds.
'''

boolq_1073 = '''
User: Passage: "Ready Player One" is a 2011 science fiction novel by Ernest Cline, set in a dystopian 2040s. It follows Wade Watts in his quest to find an Easter egg in a virtual reality game, leading to inheriting the game creator's fortune. Cline sold the rights to publish the novel in June 2010. The book was published on August 16, 2011, with an audiobook narrated by Wil Wheaton. Question: Was "Ready Player One" based on a book?
Response: True
User: Passage: Maynard's Wine Gums were introduced in 1909 by Charles Gordon Maynard, the son of Charles Riley Maynard. It took Charles Gordon Maynard some time to convince his father that the sweets didn't contain wine. Question: Did wine gums ever have wine in them?
Response: False
User: Passage: Superfecundation can result in twins with different biological fathers. Heteropaternal superfecundation occurs when two separate ova are fertilized by two different fathers. Homopaternal superfecundation leads to fraternal twins from the same father. Question: Is it possible to be pregnant with twins with two different fathers?
Response: True
User: Passage: A lone pair refers to unshared valence electrons not used in chemical bonding. A single electron doesn't count as a lone pair. Question: Does one electron count as a lone pair?
Response: False
User: Passage: The White House Medical Unit provides medical care at the White House but doesn't have a hospital. It serves White House staff, the President, Vice President, and visitors. Question: Is there a hospital in the White House?
Response: False
User: Passage: Sony Crackle offers free ad-supported video entertainment in various countries and on multiple devices, including mobile, tablet, smart TVs, desktop, and gaming consoles. It can also be accessed on American Airlines' flights and in Marriott Hotels. Question: Do you have to pay for Sony Crackle?
Response: False
User: Passage: Registering as domestic partners in California involves signing a notarized declaration listing names and addresses. The fee is $10, with an additional $23 for same-sex couples to support LGBT-specific domestic violence training and services. It's distinct from marriage or civil unions. Question: Do you have to register as a domestic partner in California?
Response: True
User: Passage: The Marine Corps has been a component of the U.S. Department of the Navy since 30 June 1834, working closely with naval forces. They also operate on land and sea-going ships and have tactical aviation squadrons embedded in Navy carrier air wings. Question: Are the Marines still a department of the Navy?
Response: True
'''

gsm8k_1061 = '''
User: Darius, Matt, and Marius are friends, who played table football. How many points did all three friends score together, if Darius scored 10 points?
Response: Darius scored 10 points. Marius scored 13 points, and Matt scored 15 points. Together, they scored 38 points.
User: A bus has a carrying capacity of 80 people. At the first pickup point, 48 people entered the bus. If there were 50 people at the next pick-up point, how many people could not take the bus because it was full?
Response: At the first pick-up point, 48 people entered, leaving 32 seats. When 50 people arrived at the next pick-up point, 18 people could not take the bus because it was full.
User: James dances twice a day for 0.5 hours each time, 4 times a week. He burned 300 calories an hour walking and loses twice as many calories per hour dancing. How many calories does he lose a week from dancing?
Response: James burns 600 calories per hour dancing. He dances for 1 hour per day, resulting in 2400 calories burned per week.
User: Happy Cattle Ranch is home to 200 cows. How many cows will the rancher own on Happy Cattle Ranch in two years if the number of cows increases by half each year?
Response: In one year, the rancher will have 300 cows. In two years, the rancher will have 450 cows.
User: Lilia has 15 peaches. She sold 10 peaches to her friends for $2 each and 4 peaches to her relatives for $1.25 each. How much did she earn after selling 14 peaches?
Response: Lilia earned $20 from her friends and $5 from her relatives, totaling $25 after selling 14 peaches.
User: Out of 150 students, the majority are girls, and 1/3 of the boys joined varsity clubs. How many of the boys did not join varsity clubs?
Response: There are 60 boys in total, and 20 of them joined varsity clubs. Therefore, 40 boys did not join varsity clubs.
User: In a game, Samanta has 8 more points than Mark, and Mark has 50% more points than Eric. How many points do Samanta, Mark, and Eric have in total if Eric has 6 points?
Response: Mark has 9 points, and Samanta has 17 points. In total, they have 32 points.
User: A bag full of sugar weighs 16 kg, and a bag full of salt weighs 30 kg. If you remove 4 kg from the combined weight of these two bags, how much do the bags weigh now?
Response: The combined weight of the bags is 46 kg. After removing 4 kg, the bags weigh 42 kg.
'''

gsm8k_1464 = '''
User: Janet hires six employees. Four are warehouse workers earning $15/hour, and two are managers earning $20/hour. She pays 10% in FICA taxes. Assuming 25 days a month and 8 hours a day, what's the total cost for wages and taxes for one month?
Response:\n - Warehouse worker monthly wage: $15/hour * 25 days * 8 hours/day = $3000.\n - Total warehouse workers' monthly wages: $3000/worker * 4 workers = $12,000.\n - Manager monthly wage: $20/hour * 25 days * 8 hours/day = $4000.\n - Total managers' monthly wages: $4000/manager * 2 managers = $8000.\n - Total monthly wages: $12,000 (warehouse workers) + $8000 (managers) = $20,000.\n - FICA taxes: 10% of $20,000 = $2000.\n - Total cost for wages and taxes: $20,000 (wages) + $2000 (taxes) = $22,000.\n Answer: $22,000
User: Ellie and Sarah got lost in a house of mirrors. Sarah saw her reflection 10 times in tall mirrors and 5 times in wide mirrors. Ellie saw her reflection 6 times in tall mirrors and 3 times in wide mirrors. Both passed through tall mirrors 3 times and wide mirrors 5 times each. How many times did they see their reflections in total?
Response:\n - Sarah's tall mirror reflections: 10 reflections * 3 passes = 30 times.\n - Sarah's wide mirror reflections: 5 reflections * 5 passes = 25 times.\n - Ellie's tall mirror reflections: 6 reflections * 3 passes = 18 times.\n - Ellie's wide mirror reflections: 3 reflections * 5 passes = 15 times.\n - Total reflections for Sarah: 30 (tall mirrors) + 25 (wide mirrors) = 55 times.\n - Total reflections for Ellie: 18 (tall mirrors) + 15 (wide mirrors) = 33 times.\n - Combined total for Sarah and Ellie: 55 + 33 = 88 times.\n Answer: 88 times
User: A curry house adjusted its pepper purchases. Previously, they bought peppers for 30 very spicy curries, 30 spicy curries, and 10 mild curries. Now, they buy peppers for 15 spicy curries and 90 mild curries. They no longer sell very spicy curries. How many fewer peppers do they buy?
Response:\n - Previous total peppers bought: (3 peppers * 30 very spicy curries) + (2 peppers * 30 spicy curries) + (1 pepper * 10 mild curries) = 160 peppers.\n - New total peppers bought: (2 peppers * 15 spicy curries) + (1 pepper * 90 mild curries) = 120 peppers.\n - Difference: 160 (previous) - 120 (new) = 40 fewer peppers.\n Answer: 40 peppers fewer
User: Peggy wants to sell her 200 records. Sammy offers $4 each for all of them. Bryan offers $6 each for half and $1 each for the rest. Peggy's records. What's the profit difference between Sammy and Bryan's deals?
Response:\n - Sammy's offer: $4/record * 200 records = $800. - Bryan's offer for the half he's interested in: $6/record * (200 records / 2) = $600.\n - Bryan's offer for the half he's not interested in: $1/record * (200 records / 2) = $100.\n - Total from Bryan: $600 + $100 = $700.\n - Difference in profit: $800 (Sammy) - $700 (Bryan) = $100.\n Answer: $100
'''

gsm8k_1233 = '''
User: Chris’s internet bill is $45 per month for 100 GB and $0.25 for every 1 GB over. His bill for this month is $65. How many GB over was Chris charged for on this bill?
Response: Let G stand for the number of GB billed, so the total bill is $0.25G + $45 = $65. Solving for G: $0.25G = $65 - $45 = $20. G = $20 / $0.25 = 80 GB. Answer: 80 GB.
User: Dakota's medical bill after a 3-day hospital stay includes $900/day for the bed, $250/hour for two specialists for 30 minutes each, and $1800 for the ambulance. What's the total?
Response: Hospital bed cost: $900/day * 3 days = $2700. Specialist cost: (30 minutes / 60 minutes/hour) * $250/hour * 2 specialists = $125. Total: $2700 + $1800 + $125 = $4625. Answer: $4625.
User: Mom bought items costing €4 each for 2 packs of bananas, €2 for pears, €6 for asparagus, and €11 for a chicken. How much money does she have left from €55?
Response: Total cost of items: €4 * 2 + €2 + €6 + €11 = €27. Money left: €55 - €27 = €28. Answer: €28.
User: L&J Productions produces 220 movies annually. Johnny TV makes 25% more movies per year. How many movies do they make together in 5 years?
Response: Johnny TV makes 25% more than L&J Productions, so 220 * 25% = 55 more movies per year. Johnny TV's total: 220 + 55 = 275 movies/year. Together in 5 years: 275 * 5 + 220 * 5 = 2475 movies. Answer: 2475.
User: The bathroom's original size is 96 sq ft with a width of 8 feet. After extending it by 2 feet on each side, what's the new area?
Response: Original length: 96 sq ft / 8 ft = 12 ft. New length: 12 ft + 2 ft = 14 ft. New width: 8 ft + 2 ft = 10 ft. New area: 14 ft * 10 ft = 140 sq ft. Answer: 140.
User: Kayla bought 12 chocolate bars and 18 soda cans. Theresa bought twice as much. How many items did Kayla buy in total?
Response: Kayla bought 12 chocolate bars and 18 soda cans. Total: 12 + 18 = 30 items. Answer: 30.
User: At a party with 1000 guests, 30% are married, 50% are single, and the rest are children. How many more married people than children are there?
Response: Married guests: 1000 * 30% = 300. Single guests: 1000 * 50% = 500. Children: 1000 - 300 - 500 = 200. Difference: 300 - 200 = 100 more married people than children. Answer: 100.
User: The flag moved up and down the 60-foot pole during the day. How far did it move?
Response: Halfway down is 60 ft / 2 = 30 ft. Total distance: 60 ft up + 30 ft down + 30 ft up + 60 ft down = 180 ft. Answer: 180 ft.
'''

multirc_2shot = '''
User: Soon after the conclusion of peace, important changes were made in legislation concerning industry and commerce, and the new freedom thus afforded produced a large number of limited liability companies. Plans were formed for building a great network of railways, partly for the purpose of developing the natural resources of the country, and partly for the purpose of increasing its power for defense and attack. The existence of serfdom was tackled boldly, taking advantage of a petition presented by the Polish landed proprietors of the Lithuanian provinces and, hoping that their relations with the serfs might be regulated in a more satisfactory way (meaning in a way more satisfactory for the proprietors), he authorized the formation of committees "for ameliorating the condition of the peasants," and laid down the principles on which the amelioration was to be effected. This step had been followed by one even more significant. Without consulting his ordinary advisers, Alexander ordered the Minister of the Interior to send a circular to the provincial governors of European Russia (serfdom was rare in other parts), containing a copy of the instructions forwarded to the Governor-General of Lithuania, praising the supposed generous, patriotic intentions of the Lithuanian landed proprietors, and suggesting that perhaps the landed proprietors of other provinces might express a similar desire. The hint was taken: in all provinces where serfdom existed, emancipation committees were formed. The emancipation was not merely a humanitarian question capable of being solved instantaneously by imperial ukase. It contained very complicated problems, deeply affecting the economic, social and political future of the nation. Alexander had to choose between the different measures recommended to him and decide if the serfs would become agricultural laborers dependent economically and administratively on the landlords or if the serfs would be transformed into a class of independent communal proprietors. The emperor gave his support to the latter project, and the Russian peasantry became one of the last groups of peasants in Europe to shake off serfdom. The architects of the emancipation manifesto were Alexander's brother Konstantin, Yakov Rostovtsev, and Nikolay Milyutin. On 3 March 1861, 6 years after his accession, the emancipation law was signed and published. According to the passage, answer all the following multiple-choice questions:
    Question0: "What contained a very complicated problems that affected the economic, social, and political future of Russia?"  Choices: A. The emancipation B. The railway plans C. The emancipation of the serfs D. A peace treaty 
    Question1: "What significant event followed after a petition by Polish landed proprietors was presented to Tsar Alexander?"  Choices: A. Alexander ordered his Minister of the Interior to send a message to the governors of European Russia praising the Polish landed proprietors idea to emancipate the serfs B. Alexander ordered the Minister of the Interior to send a circular to the provincial governors of European Russia C. Plans were formed for building a great network of railways D. Alexander died E. Secret service 
    Question2: "When were plans formed for building a great network of railways?"  Choices: A. Soon after the conclusion of peace B. During a war C. On 3 March 1861 
    Question3: "Outside of posing the humanitarian question, what else did the emancipation serve?"  Choices: A. The architects of the emancipation manifesto B. The economic, social and political future of the nation C. Issues deeply affecting the economic, social and political future of the nation D. The existence of serfdom 
    Question4: "When was the existence of serfdom tackled?"  Choices: A. 6 years after his accession B. Soon after the conclusion of peace C. In March, 1931 
    Question5: "Who authorized the formation of committees "for ameliorating the condition of the peasants"?"  Choices: A. The Governor-General of Lithuania B. The Russian peasantry C. Tsar Alexander D. Alexander 
    Question6: "Which idea for the emancipation of the serfs did Alexander lend his support to?"  Choices: A. Transforming the serfs into a class of independent communal proprietors B. Ther serfs becomin dependent laborers C. The serfs being transformed into a class of independent communal proprietors D. The serfs leaving Russia forever 
Response: Question0: AC  Question1: AB  Question2: A  Question3: BC  Question4: B  Question5: CD  Question6: AC

User: Even though electronic espionage may cost U.S. firms billions of dollars a year, most aren't yet taking precautions, the experts said. By contrast, European firms will spend $150 million this year on electronic security, and are expected to spend $1 billion by 1992. Already many foreign firms, especially banks, have their own cryptographers, conference speakers reported. Still, encrypting corporate communications is only a partial remedy. One expert, whose job is so politically sensitive that he spoke on condition that he wouldn't be named or quoted, said the expected influx of East European refugees over the next few years will greatly increase the chances of computer-maintenance workers, for example, doubling as foreign spies. Moreover, he said, technology now exists for stealing corporate secrets after they've been "erased" from a computer's memory. He said that Oliver North of Iran-Contra notoriety thought he had erased his computer but that the information was later retrieved for congressional committees to read. No personal computer, not even the one on a chief executive's desk, is safe, this speaker noted. W. Mark Goode, president of Micronyx Inc., a Richardson, Texas, firm that makes computer-security products, provided a new definition for Mikhail Gorbachev's campaign for greater openness, known commonly as glasnost. Under Mr. Gorbachev, Mr. Goode said, the Soviets are openly stealing Western corporate communications. He cited the case of a Swiss oil trader who recently put out bids via telex for an oil tanker to pick up a cargo of crude in the Middle East. Among the responses the Swiss trader got was one from the Soviet national shipping company, which hadn't been invited to submit a bid. The Soviets' eavesdropping paid off, however, because they got the contract.  According to the passage, answer all the following multiple-choice questions:
    Question0: "Who did the Soviets contract with to pick up a cargo of crude oil in the Middle East?"  Choices: A. Dubai oil trader B. Israeli Oil Trader C. Swiss D. Iran E. A Swiss oil trader F. Iraqi oil trader G. Country bordering Switzerland H. North Korea 
    Question1: "What is the full name of the man who claimed that the Soviets are openly stealing Western corporate communications?"  Choices: A. Gorbachev B. W. Mark Goode C. Mark Goode D. Oliver North E. Richardson F. Mr. Gorbachev G. Mikhail Gorbachev 
    Question2: "Is the step that foreign banks have begun to apply likely to solve the problem completely?"  Choices: A. Yes B. Impossible C. Maybe 
    Question3: "Are Europeans spending more or less to combat electronic espionage than the U.S.?"Choices: A. About the same B. More C. A lot more D. Less 
Response: Question0: CE  Question1: B  Question2: C  Question3: B
'''

multirc_distilled_2shot = '''
User: After peace, there were changes in industry and commerce legislation, leading to the formation of many limited liability companies. Railways were planned for resource development and defense. Serfdom was addressed following a petition by Polish landowners in Lithuania, leading to committees for peasants' betterment and eventual emancipation. Alexander supported turning serfs into independent communal proprietors. Emancipation law was signed in 1861.
    Question0: "What complex issue affected Russia's future?" Answers: A. The emancipation B. The railway plans C. The emancipation of serfs D. A peace treaty 
    Question1: "What significant event followed a petition by Polish landowners?" Answers: A. Alexander praised the idea to emancipate serfs B. Alexander ordered circulars to governors C. Railway plans were formed D. Alexander died E. Secret service 
    Question2: "When were railway plans made?" Answers: A. After peace B. During a war C. On March 3, 1861 
    Question3: "Aside from humanitarian concerns, what did emancipation serve?" Answers: A. The architects of the emancipation manifesto B. Economic, social, and political future C. Issues affecting the nation D. The existence of serfdom 
    Question4: "When was serfdom addressed?" Answers: A. 6 years after accession B. After peace C. In March 1931 
    uestion5: "Who authorized committees for peasants' betterment?" Answers: A. Governor-General of Lithuania B. Russian peasantry C. Tsar Alexander D. Alexander 
    Question6: "Which idea for serf emancipation did Alexander support?" Answers: A. Independent communal proprietors B. Serfs as dependent laborers C. Serfs as independent communal proprietors D. Serfs leaving Russia 
Response: Question0: AC  Question1: AB  Question2: A  Question3: BC  Question4: B  Question5: CD  Question6: AC

User: Electronic espionage costs U.S. firms billions, but most don't take precautions. European firms spend on electronic security, yet encrypting communications is only partial protection. An expert mentioned an expected influx of East European refugees doubling as spies. Technology allows stealing erased corporate secrets. No personal computer is safe. Mark Goode, president of Micronyx Inc., mentioned the Soviets openly stealing Western corporate communications, citing a case involving a Swiss oil trader and the Soviet national shipping company.
    Question0: "Who did the Soviets contract with for crude oil transport in the Middle East?" Answers: A. Dubai oil trader B. Israeli Oil Trader C. Swiss D. Iran E. A Swiss oil trader F. Iraqi oil trader G. Country bordering Switzerland H. North Korea 
    Question1: "What is the full name of the expert claiming Soviets steal Western corporate communications?" Answers: A. Gorbachev B. W. Mark Goode C. Mark Goode D. Oliver North E. Richardson F. Mr. Gorbachev G. Mikhail Gorbachev 
    Question2: "Will the foreign banks' steps likely completely solve the problem?" Answers: A. Yes B. Impossible C. Maybe 
    Question3: "Do Europeans spend more or less on countering electronic espionage compared to the U.S.?" Answers: A. About the same B. More C. A lot more D. Less 
Response: Question0: CE  Question1: B  Question2: C  Question3: B
'''
distilled = '''
User: Darius, Matt, and Marius are friends, who played table football. During all the games they played, Marius scored 3 points more than Darius, and Darius scored 5 points less than Matt. How many points did all three friends score together, if Darius scored 10 points?
Response: Darius: 10 points, Marius: 13 points, Matt: 15 points. Total: 38 points.

User: A bus has a carrying capacity of 80 people. At the first pickup point, the number of people who entered the bus was 3/5 of its carrying capacity. If there were 50 people at the next pick-up point, how many people could not take the bus because it was full?
Response: First pickup: 48 people. Available seats: 32. People who couldn't take the bus: 18.

User: James takes up dancing for fitness. He loses twice as many calories per hour as he did when he was walking. He dances twice a day for .5 hours each time and he does this 4 times a week. He burned 300 calories an hour walking. How many calories does he lose a week from dancing?
Response: Calorie burn per hour dancing: 600 calories. Daily burn: 300 calories. Weekly burn: 1200 calories.

User: Happy Cattle Ranch is home to 200 cows. Every year, the mother cows have enough calves that the number of cows the rancher owns rises by half the number of cows living on the ranch. If the rancher does not buy or sell any cows, how many cows will the rancher own on Happy Cattle Ranch in two years?
Response: After 1 year: 300 cows. After 2 years: 450 cows.

User: Lilia has 15 peaches. She sold 10 peaches to her friends for $2 each, while 4 other peaches were sold to her relatives for $1.25 each, and she left the other one for herself. How much did she earn after selling 14 peaches?
Response: Earned from friends: $20. Earned from relatives: $5. Total earned from 14 peaches: $25.

User: Out of the 150 students, 60% are girls and the rest are boys. Only 1/3 of the boys joined varsity clubs. How many of the boys did not join varsity clubs?
Response: Boys: 60 boys. Boys in varsity clubs: 20. Boys not in varsity clubs: 40.

User: In a game, Samanta has 8 more points than Mark, and Mark has 50% more points than Eric. Eric has 6 points. How many points do Samanta, Mark, and Eric have in total?
Response: Eric: 6 points, Mark: 9 points, Samanta: 17 points. Total: 32 points.

User: A bag full of sugar weighs 16 kg. A bag full of salt weighs 30 kg. If you remove 4 kg from the combined weight of these two bags, how much do the bags now weigh?
Response: Combined weight: 46 kg. After removing 4 kg: 42 kg.
'''
multirc = '''
User: Passage: "Iwasaki's careful respect for the Long Beach program and its lawyers earned him the political capital he needed to complete his takeover in a matter of weeks. "The Long Beach program had strong support in the community so in a situation like that, one has to recognize that it's not like a takeover where I have all the answers and I know best," Iwasaki said. "The people in the community who are working there have the contacts and the knowledge that will allow service to continue." Things have gone less smoothly across town. There, Dudovitz, a longtime poverty lawyer and executive director of the San Fernando Valley's 36-year-old legal aid program, continues to struggle with his hostile takeover of the neighboring San Gabriel-Pomona Valleys service area one year after it was accomplished. On the bright side, Dudovitz has extended his respected program to clients in the San Gabriel-Pomona Valley, and he now operates on a much larger budget, $6.5 million last year. However, his clash with the old San Gabriel program resulted in litigation, bitter feelings and a mission that some say is not clearly focused on serving poor people. "It was a difficult situation that was probably mishandled by everyone," a longtime observer of the public interest community said of the San Fernando Valley-San Gabriel-Pomona Valley merger. "There are very few people who come out as the heroes. Personalities got involved when they shouldn't have. Things were said that caused bad feelings and couldn't be unsaid." Iwasaki's merger with the smaller, 48-year-old Long Beach program was friendly and fast, and no one - not even Long Beach board members - lost a job. When it was over, Iwasaki had $1 million more in federal dollars and two new offices. Long Beach clients regained services they had lost years ago when federal budget cuts and dwindling grants reduced the staff of 15 lawyers to five and cut immigration and consumer law programs. Iwasaki said, "[I judged the transition] better than I could have hoped for." "
    Question1: "What is the difference in the ages of the Long Beach and San Fernando Valley programs?"
    Choices: A. 8 B. 48 years C. 12 years D. 30 year E. 48 F. Weeks G. 36 H. Can not figure out as one has 37 year and other has no age limit I. 36 years J. Eight K. 6.5 years 
Response: Answer1: ABCHJ

User: Even though electronic espionage may cost U.S. firms billions of dollars a year, most aren't yet taking precautions, the experts said. By contrast, European firms will spend $150 million this year on electronic security, and are expected to spend $1 billion by 1992. Already many foreign firms, especially banks, have their own cryptographers, conference speakers reported. Still, encrypting corporate communications is only a partial remedy. One expert, whose job is so politically sensitive that he spoke on condition that he wouldn't be named or quoted, said the expected influx of East European refugees over the next few years will greatly increase the chances of computer-maintenance workers, for example, doubling as foreign spies. Moreover, he said, technology now exists for stealing corporate secrets after they've been "erased" from a computer's memory. He said that Oliver North of Iran-Contra notoriety thought he had erased his computer but that the information was later retrieved for congressional committees to read. No personal computer, not even the one on a chief executive's desk, is safe, this speaker noted. W. Mark Goode, president of Micronyx Inc., a Richardson, Texas, firm that makes computer-security products, provided a new definition for Mikhail Gorbachev's campaign for greater openness, known commonly as glasnost. Under Mr. Gorbachev, Mr. Goode said, the Soviets are openly stealing Western corporate communications. He cited the case of a Swiss oil trader who recently put out bids via telex for an oil tanker to pick up a cargo of crude in the Middle East. Among the responses the Swiss trader got was one from the Soviet national shipping company, which hadn't been invited to submit a bid. The Soviets' eavesdropping paid off, however, because they got the contract.  According to the passage, answer all the following multiple-choice questions:
    Question0: "Who did the Soviets contract with to pick up a cargo of crude oil in the Middle East?"  
    Choices: A. Dubai oil trader B. Israeli Oil Trader C. Swiss D. Iran E. A Swiss oil trader F. Iraqi oil trader G. Country bordering Switzerland H. North Korea 
Response: Question0: CE

User: Passage: "Honours and legacy In 1929, Soviet writer Leonid Grossman published a novel The d'Archiac Papers, telling the story of Pushkin's death from the perspective of a French diplomat, being a participant and a witness of the fatal duel. The book describes him as a liberal and a victim of the Tsarist regime. In Poland the book was published under the title Death of the Poet. In 1937, the town of Tsarskoye Selo was renamed Pushkin in his honour. There are several museums in Russia dedicated to Pushkin, including two in Moscow, one in Saint Petersburg, and a large complex in Mikhaylovskoye. Pushkin's death was portrayed in the 2006 biographical film Pushkin: The Last Duel. The film was directed by Natalya Bondarchuk. Pushkin was portrayed onscreen by Sergei Bezrukov. The Pushkin Trust was established in 1987 by the Duchess of Abercorn to commemorate the creative legacy and spirit of her ancestor and to release the creativity and imagination of the children of Ireland by providing them with opportunities to communicate their thoughts, feelings and experiences. A minor planet, 2208 Pushkin, discovered in 1977 by Soviet astronomer Nikolai Stepanovich Chernykh, is named after him. A crater on Mercury is also named in his honour. MS Alexandr Pushkin, second ship of the Russian Ivan Franko class (also referred to as "poet" or "writer" class). Station of Tashkent metro was named in his honour. The Pushkin Hills and Pushkin Lake were named in his honour in Ben Nevis Township, Cochrane District, in Ontario, Canada. UN Russian Language Day, established by the United Nations in 2010 and celebrated each year on 6 June, was scheduled to coincide with Pushkin's birthday. "
    Question0: "What is the name of the novel that was later published in Poland under the title "Death of the Poet"?"
    Choices: A. The poet papers B. "The d'Archiac Papers."
Response: Answer0: B

User: Passage: "If you beat a dog in Schuylkill County, you'll probably get a $100 fine. If you repeatedly beat a woman, you'll probably get the same fine. In 2001, county judges heard 98 Protection From Abuse cases, finding the defendant guilty in 48 percent of those cases, either after a hearing or through a technical violation or plea. Of those found guilty, the majority were ordered to pay court costs, plus a $100 fine. No defendants were ordered to pay more than a $250 fine for violating the court order. In 27 percent of the cases, the charges were dismissed or the defendant was found not guilty. In the rest of the cases, charges were withdrawn or the matter is not yet resolved. Sarah T. Casey, executive director of Schuylkill Women in Crisis, finds it disturbing that in most cases, the fine for violating a PFA is little more than the fine someone would get for cruelty and abuse toward an animal. "In most of the counties surrounding Schuylkill County, the penalties given for indirect criminal contempt are much stiffer than those in Schuylkill County," Casey said. "What kind of message are we sending those who repeatedly violate Protection From Abuse orders? That it's OK to abuse women in Schuylkill County, because you'll only get a slap on the wrist?" Under state law, the minimum fine for contempt of a PFA is $100; the maximum fine is $1,000 and up to six months in jail. Like others who are familiar with how the county's legal system does and doesn't work for victims of domestic violence, Casey believes some changes are in order. Valerie West, a manager/attorney with Mid-Penn Legal Services, with offices in Pottsville and Reading, regularly handles domestic violence cases. She finds fault with the local requirement that a custody order must be established within 30 days after a PFA is filed. West said she feels a custody order should be allowed to stand for the full term of the PFA - up to 18 months - as it does in many other counties in the state. "It places an undue burden on the plaintiff, in terms of cost, finding legal representation and facing their abuser - not to mention a further burden on the system to provide those services," West said. "It may be difficult for the parties to reach an agreement so soon after violence has occurred. "
    Question9: "If you beat a dog and woman in Schuylkill County how much of a fine will you need to pay?"
    Choices: A. $1,000 B. $250 C. $100 D. $48
Response: Answer9: C    
'''
prompts = [
    # prompt_381,
    # prompt_731,
    # prompt_1368,
    # prompt_615,
    # prompt_557,
    # prompt_1643,
    # boolq,
    # prompt_half,
    # prompt_quater,
    # prompt_1126,
    # distill_prompt,
    # prompt_5,
    # prompt_3,
    # prompt_1214,
    # boolq_1073,
    # gsm8k_1061,
    # gsm8k_1464,
    # gsm8k_1233,
    # multirc_2shot,
    # multirc_distilled_2shot,
    distilled,
    # multirc
]

for prompt in prompts:
    print(num_tokens_from_string(prompt, "gpt-3.5-turbo"))