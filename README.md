Zavrsni rad 2025. Suparnički napadi na modele za klasifikaciju slika
Upute za pokretanje

Potrebni paketi: torch, torchvision, matplotlib, numpy

1. Pozicionirati se u fgsm_final direktorij:
  cd Adversarial-attacks-image-classification/main_workspace/fgsm_final

2. Pokrenuti željeni program: (nije nužno pokrenuti 2.1. i 2.2. prije 2.3., može se testirati bilo kojim redosljedom)

  2.1. Pokretanje sgd algoritma (treniranje potpunopovezanog modela na skupu podataka mnist):
    python test_sgd.py

  2.2. Testiranje resnet-20 modela na CIFAR-100 ispitnom skupu:
    python test_cifar_model.py
  
  2.3. Pokretanje napada
    python testAttack2.py STRATEGIJA [--targeted | --untargeted] [--same | --diff] [--graph | --epsilon EPSILON]

    STRATEGIJA: mnist ili cifar100
    --targeted: ciljani napad
    --untargeted: neciljani napad
    --same: koristi ciljnu klasu iz iste nadklase kao i inicijalna klasifikacija (samo za cifar100 + --targeted)
    --diff: koristi ciljnu klasu iz različite nadklase nego inicijalna klasifikacija (samo za cifar100 + --targeted)
    --graph: generira graf uspješnosti za više epsilon vrijednosti
    --epsilon EPSILON: pokreće napad za jednu epsilon vrijednost i viziulizira nekoliko suparničkih primjera za tu vrijednost

    primjeri:
      python testAttack2.py mnist --targeted --graph
      python testAttack2.py mnist --untargeted --epsilon 0.2
      python testAttack2.py cifar100 --targeted --same --graph
      python testAttack2.py cifar100 --targeted --diff --epsilon 0.3
      python testAttack2.py cifar100 --untargeted --graph

3. Pregled rezultata
   Rezultati napada dobiveni s 2.3. bit će spremljeni u attack_data_NEW direktorij
   Prethodno dobiveni rezultati napada od kojih je dio uključen u tekst završnog rada nalaze se u direktoriju attack_data
   U direktoriju attack_data_denorm se nalaze rezultati koji su dobiveni tako da se perturbacija dodala na denormalizirane slike umjesto normalizirane (što nije dobro)
   Grafovi za sgd algoritam nalaze se u direktoriju learning_data
   
