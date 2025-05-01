from poker import Deck, pcg64, ThreadPool, probability_of_winning
from random import randint

def main():
    pcg = pcg64(randint(0, 2**64 - 1))
    deck = Deck.create_full_deck()
    player_deck = deck.pop_random_cards(pcg, 2)
    table_deck = deck.pop_random_cards(pcg, 3)
    tp = ThreadPool(5)
    num_players = 2
    num_simulations = 1_000_000
    print("Player's hand:", player_deck)
    print("Table cards:", table_deck)
    print("Number of players:", num_players)
    print("Number of simulations:", num_simulations)
    while True:
        probability = probability_of_winning(player_deck, table_deck, num_simulations, num_players, tp)
        print(f"Probability of winning: {probability:.2%}")

if __name__ == "__main__":
    main()